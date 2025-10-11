const DEFAULT_SETTINGS = {
	serverUrl: 'http://127.0.0.1:8000',
	minSeverityToHide: 2, // 2=악성 차단, 1=약간 악성 이상 차단
	action: 'hide', // hide | blur | delete
	showBadge: true,
};

async function getSettings() {
	return new Promise((resolve) => {
		chrome.storage.local.get(DEFAULT_SETTINGS, (values) => resolve(values));
	});
}

async function classify(texts) {
	const { serverUrl } = await getSettings();
	const url = `${serverUrl.replace(/\/$/, '')}/predict`;
	try {
		const res = await fetch(url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
			body: JSON.stringify({ texts }),
		});
		if (!res.ok) throw new Error(`HTTP ${res.status}`);
		return await res.json();
	} catch (e) {
		console.warn('Classification request failed:', e);
		return { labels: [], probs: [], label_names: {} };
	}
}

async function sendTrainingData(text, label, userId = 'anonymous', useTemp = true) {
    const { serverUrl } = await getSettings();
    const url = `${serverUrl.replace(/\/$/, '')}/training-data${useTemp ? '?temp=1' : ''}`;
	try {
		const res = await fetch(url, {
			method: 'POST',
			headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text, label, user_id: userId }),
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

// onSuspend에서 임시 데이터 자동 삭제는 비활성화 (MV3 수면으로 인해 캐시가 즉시 사라지는 문제 방지)

chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
    if (message?.type === 'classify') {
        // 캐시 조회 → 미스만 분류 → 캐시 저장
        lookupCachedLabels(message.texts).then(async (cached) => {
            const out = new Array(message.texts.length).fill(null);
            const missIdxs = [];
            message.texts.forEach((t, i) => {
                const lbl = cached?.labels?.[i];
                if (lbl === 0 || lbl === 1 || lbl === 2) out[i] = lbl; else missIdxs.push(i);
            });
            if (missIdxs.length === 0) { sendResponse({ labels: out, probs: [], label_names: {} }); return; }
            const missTexts = missIdxs.map(i => message.texts[i]);
            const predicted = await classify(missTexts);
            missIdxs.forEach((i, k) => { out[i] = predicted.labels?.[k] ?? 0; });
            // 분류 결과 temp 캐시 저장
            missIdxs.forEach((i, k) => {
                const lbl = predicted.labels?.[k];
                if (lbl === 0 || lbl === 1 || lbl === 2) sendTrainingData(message.texts[i], lbl, 'cache', true).catch(()=>{});
            });
            sendResponse({ labels: out, probs: predicted.probs, label_names: predicted.label_names });
        });
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
		sendTrainingData(message.text, message.label, message.userId).then(sendResponse);
		return true;
	}
	return false;
});

