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
  const isExternal = (() => {
    try { const u = new URL(serverUrl); return (u.port === '3000') || /223\.194\.46\.69/.test(u.hostname); } catch { return /:3000$/.test(serverUrl); }
  })();
  const base = (serverUrl || '').replace(/\/$/, '');
  const url = isExternal ? `${base}/api/predict` : `${base}/predict`;
  try {
    if (isExternal) {
      // 외부 서버는 단건 응답({ ok, data:{label,...} })을 반환하므로 텍스트별로 순차 호출하여 배열로 변환
      const labels = [];
      for (const t of texts) {
        try {
          const res = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ texts: [t] })
          });
          if (!res.ok) throw new Error(`HTTP ${res.status}`);
          const j = await res.json();
          const lbl = j?.data?.label;
          labels.push((lbl === 0 || lbl === 1 || lbl === 2) ? lbl : 0);
        } catch {
          labels.push(0);
        }
      }
      return { labels, probs: [], label_names: {} };
    } else {
      const res = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ texts })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      return await res.json();
    }
  } catch (e) {
    console.warn('Classification request failed:', e);
    return { labels: [], probs: [], label_names: {} };
  }
}

async function sendTrainingData(text, label, userId = 'anonymous', useTemp = false) {
  const { serverUrl } = await getSettings();
  // 외부 서버 모드에서도 학습 데이터는 로컬 서버(127.0.0.1:8000)에 저장
  let base = (serverUrl || '').replace(/\/$/, '');
  try {
    const u = new URL(serverUrl || '');
    if (u.port === '3000' || /223\.194\.46\.69/.test(u.hostname)) {
      base = 'http://127.0.0.1:8000';
    }
  } catch {
    if (/:3000$/.test(String(serverUrl||''))) {
      base = 'http://127.0.0.1:8000';
    }
  }
  const url = `${base}/training-data${useTemp ? '?temp=1' : ''}`;
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
      const isExternal = (() => {
        try { const u = new URL(settings.serverUrl || ''); return (u.port === '3000') || /223\.194\.46\.69/.test(u.hostname); } catch { return /:3000$/.test(String(settings.serverUrl||'')); }
      })();

      const originalTexts = message.texts || [];
      // 1) 모델 입력용 마스크 전처리
      const maskedTexts = originalTexts.map(t => applyMaskToText(t, masks));

      // 2) 캐시 조회(마스크된 텍스트 기준) - 외부 서버 모드에서는 생략
      const cached = isExternal ? { labels: [] } : await lookupCachedLabels(maskedTexts);

      // 3) 캐시 미스만 서버 분류
      const out = new Array(originalTexts.length).fill(null);
      if (isExternal) {
        const predicted = await classify(maskedTexts);
        maskedTexts.forEach((_, i) => { out[i] = predicted.labels?.[i] ?? 0; });
      } else {
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
          // 4) (삭제됨) 분류 결과를 임시 캐시로 전송/저장하는 기능은 비활성화되었습니다.
        }
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
    // 임시 저장 비활성화: 항상 영구 저장으로 전송
    sendTrainingData(message.text, message.label, message.userId, false).then(sendResponse);
    return true;
  }

  return false;
});
