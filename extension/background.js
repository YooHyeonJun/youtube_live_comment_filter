// === 변경된 background.js (발췌; 파일 전체 대체 권장) ===

// 새 기본 설정: rules, masks 추가
const DEFAULT_SETTINGS = {
  serverUrl: 'http://127.0.0.1:8000',
  minSeverityToHide: 2,       // 2=악성 차단, 1=약간 악성 이상 차단
  action: 'hide',             // hide | blur | delete
  showBadge: true,
  rules: [],                  // [{ term: '시발', min: 2 }, { term: '이재명', min: 1 }]
  masks: []                   // ['시발', '욕설A', ...]
};

async function getSettings() {
  return new Promise(resolve => {
    chrome.storage.local.get(DEFAULT_SETTINGS, values => resolve(values));
  });
}

// 공백/구두점 경계까지 엄격할 필요가 있다면 정규식 개선 가능
function normalize(s){ return (s || '').toString(); }

// --- 마스크 전처리: 모델 입력 전 불필요 단어 제거 ---
function applyMaskToText(text, masks) {
  if (!text || !Array.isArray(masks) || masks.length === 0) return text;
  let out = text;
  for (const m of masks) {
    if (!m) continue;
    // 단순 포함 제거(단어 경계가 필요하면 \b 사용 고려)
    const esc = m.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    out = out.replace(new RegExp(esc, 'g'), '');
  }
  return out.trim().replace(/\s{2,}/g, ' ');
}

// --- 룰 적용: 특정 단어 포함 시 최소 심각도 강제 ---
function applyRuleFloorToLabel(originalText, modelLabel, rules) {
  if (!originalText || !Array.isArray(rules) || rules.length === 0) return modelLabel ?? 0;
  const src = normalize(originalText);
  let floor = 0;
  for (const r of rules) {
    if (!r?.term) continue;
    const term = normalize(r.term);
    const min = Number(r.min ?? 0);
    if (!term) continue;
    if (src.includes(term)) {
      if (min > floor) floor = min;
    }
  }
  // 최소 0은 그대로(상향 없음). 그 외엔 floor를 하한선으로 적용.
  return Math.max(modelLabel ?? 0, floor);
}

async function classify(texts) {
  const { serverUrl } = await getSettings();
  const url = `${serverUrl.replace(/\/$/, '')}/predict`;
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.warn('Classification request failed:', e);
    return { labels: [], probs: [], label_names: {} };
  }
}

async function sendTrainingData(text, label, userId = 'anonymous', useTemp = false) {
  const { serverUrl } = await getSettings();
  const url = `${serverUrl.replace(/\/$/, '')}/training-data${useTemp ? '?temp=1' : ''}`;
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, label, user_id: userId })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    console.warn('Training data send failed:', e);
    return { success: false, message: e.message };
  }
}

async function lookupCachedLabels(texts) {
  const { serverUrl } = await getSettings();
  const url = `${serverUrl.replace(/\/$/, '')}/training-data/lookup`;
  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ texts })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return await res.json();
  } catch (e) {
    return { labels: [] };
  }
}

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  if (message?.type === 'classify') {
    (async () => {
      const settings = await getSettings();
      const rules = Array.isArray(settings.rules) ? settings.rules : [];
      const masks = Array.isArray(settings.masks) ? settings.masks : [];

      const originalTexts = message.texts || [];
      // 1) 모델 입력용 마스크 전처리
      const maskedTexts = originalTexts.map(t => applyMaskToText(t, masks));

      // 2) 캐시 조회(마스크된 텍스트 기준)
      const cached = await lookupCachedLabels(maskedTexts);

      // 3) 캐시 미스만 서버 분류
      const out = new Array(originalTexts.length).fill(null);
      const missIdxs = [];
      maskedTexts.forEach((t, i) => {
        const lbl = cached?.labels?.[i];
        if (lbl === 0 || lbl === 1 || lbl === 2) out[i] = lbl; else missIdxs.push(i);
      });
      if (missIdxs.length > 0) {
        const missTexts = missIdxs.map(i => maskedTexts[i]);
        const predicted = await classify(missTexts);
        missIdxs.forEach((i, k) => {
          out[i] = predicted.labels?.[k] ?? 0;
        });
        // 4) 분류 결과 temp 캐시 저장(키=마스크된 텍스트)
        missIdxs.forEach((i, k) => {
          const lbl = out[i];
          if (lbl === 0 || lbl === 1 || lbl === 2) {
            sendTrainingData(maskedTexts[i], lbl, 'cache', true).catch(() => {});
          }
        });
      }

      // 5) 룰 최소 심각도 적용(원문 기준)
      const ruled = out.map((lbl, i) => applyRuleFloorToLabel(originalTexts[i], lbl, rules));

      sendResponse({ labels: ruled, probs: [], label_names: {} });
    })();
    return true; // async
  }

  if (message?.type === 'getSettings') {
    getSettings().then(sendResponse);
    return true;
  }

  if (message?.type === 'incCounter') {
    chrome.action.getBadgeText({}, async (t) => {
      const curr = Number(t || '0') || 0;
      const next = String(curr + (message.by || 1));
      chrome.action.setBadgeText({ text: next });
      chrome.action.setBadgeBackgroundColor({ color: '#d9534f' });
      sendResponse({ ok: true, count: next });
    });
    return true;
  }

  if (message?.type === 'sendTrainingData') {
    const useTemp = message.useTemp === true; // 명시 요청 시에만 temp 사용
    sendTrainingData(message.text, message.label, message.userId, useTemp).then(sendResponse);
    return true;
  }

  return false;
});
