const DEF = { enabled: true, minSeverityToHide: 2, action: 'hide', showBadge: true, trainingMode: false, serverUrl: 'http://127.0.0.1:8000' };
function getValues() { return new Promise(r => chrome.storage.local.get(DEF, r)); }
function setValues(v) { return new Promise(r => chrome.storage.local.set(v, r)); }

async function getTrainingStats() {
	const { serverUrl } = await getValues();
	try {
		const response = await fetch(`${serverUrl}/training-data/stats`);
		if (response.ok) {
			return await response.json();
		}
	} catch (e) {
		console.warn('Failed to get training stats:', e);
	}
	return { total_samples: 0, label_distribution: { 정상: 0, 약간_악성: 0, 악성: 0 } };
}

async function getTrainingFiles() {
    const { serverUrl } = await getValues();
    const res = await fetch(`${serverUrl}/training-data/files`);
    if (!res.ok) return { files: [] };
    return await res.json();
}

async function deleteAllTrainingData() {
    const { serverUrl } = await getValues();
    const res = await fetch(`${serverUrl}/training-data/all`, { method: 'DELETE' });
    return res.ok;
}

async function deleteTempOnly() {
    const { serverUrl } = await getValues();
    const res = await fetch(`${serverUrl}/training-data/temp`, { method: 'DELETE' });
    return res.ok;
}

async function startRetraining() {
	const { serverUrl } = await getValues();
	try {
		const response = await fetch(`${serverUrl}/model/retrain`, { method: 'POST' });
		return await response.json();
	} catch (e) {
		console.warn('Failed to start retraining:', e);
		return { success: false, message: e.message };
	}
}

async function reloadModel() {
	const { serverUrl } = await getValues();
	try {
		const response = await fetch(`${serverUrl}/model/reload`, { method: 'POST' });
		return await response.json();
	} catch (e) {
		console.warn('Failed to reload model:', e);
		return { success: false, message: e.message };
	}
}

async function getTrainingStatus() {
	const { serverUrl } = await getValues();
	try {
		const response = await fetch(`${serverUrl}/model/training-status`);
		if (response.ok) {
			return await response.json();
		}
	} catch (e) {
		console.warn('Failed to get training status:', e);
	}
	return { is_training: false, progress: 0, message: '', error: null };
}

async function updateTrainingStats() {
    const stats = await getTrainingStats();
    const dist = stats.label_distribution || {};
    const normal = (dist['정상'] ?? dist['label_0'] ?? dist['0'] ?? 0);
    const mild   = (dist['약간 악성'] ?? dist['약간_악성'] ?? dist['label_1'] ?? dist['1'] ?? 0);
    const bad    = (dist['악성'] ?? dist['label_2'] ?? dist['2'] ?? 0);
    document.getElementById('dataCount').textContent = stats.total_samples || (normal + mild + bad) || 0;
    document.getElementById('normalCount').textContent = normal;
    document.getElementById('mildCount').textContent = mild;
    document.getElementById('maliciousCount').textContent = bad;
}

async function updateTrainingStatus() {
	const status = await getTrainingStatus();
	const progressDiv = document.getElementById('trainingProgress');
	const progressFill = document.getElementById('progressFill');
	const progressText = document.getElementById('progressText');
	const retrainBtn = document.getElementById('retrainModel');
	
	if (status.is_training) {
		progressDiv.style.display = 'block';
		progressFill.style.width = `${status.progress}%`;
		progressText.textContent = status.message || '학습 중...';
		retrainBtn.disabled = true;
		retrainBtn.textContent = '재학습 중...';
	} else {
		progressDiv.style.display = 'none';
		retrainBtn.disabled = false;
		retrainBtn.textContent = '재학습 시작';
		
		if (status.error) {
			progressDiv.style.display = 'block';
			progressText.textContent = `오류: ${status.error}`;
			progressText.style.color = '#d9534f';
		} else if (status.message && status.message.includes('완료')) {
			progressDiv.style.display = 'block';
			progressText.textContent = status.message;
			progressText.style.color = '#5bc0de';
		}
	}
}

let BUSY = false;

async function initPopup() {
	const v = await getValues();
	const min = document.getElementById('min');
	const act = document.getElementById('act');
	const badge = document.getElementById('badge');
	const count = document.getElementById('count');
	const trainingMode = document.getElementById('trainingMode');
    const enabled = document.getElementById('enabled');
	
	min.value = String(v.minSeverityToHide);
	act.value = v.action;
	badge.checked = !!v.showBadge;
	trainingMode.checked = !!v.trainingMode;
    if (enabled) enabled.checked = v.enabled !== false;
	
	chrome.action.getBadgeText({}, t => count.textContent = t || '0');
	
	min.onchange = () => setValues({ minSeverityToHide: Number(min.value) });
	act.onchange = () => setValues({ action: act.value });
	badge.onchange = () => setValues({ showBadge: badge.checked });
    trainingMode.onchange = (e) => { e.preventDefault(); e.stopPropagation(); setValues({ trainingMode: trainingMode.checked }); };
    // 토글 클릭/키입력 버블링 차단 (재학습 버튼 오작동 방지)
    trainingMode.addEventListener('click', (e)=>{ e.stopPropagation(); }, true);
    trainingMode.addEventListener('keydown', (e)=>{ e.stopPropagation(); }, true);
    if (enabled) enabled.onchange = () => setValues({ enabled: enabled.checked });
	
	// 학습 데이터 통계 업데이트
	await updateTrainingStats();
	
	// 재학습 상태 업데이트
	await updateTrainingStatus();
	
	// 재학습 버튼 이벤트
    const retrainBtn = document.getElementById('retrainModel');
	if (retrainBtn) {
        retrainBtn.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (BUSY) return;
            BUSY = true;
			retrainBtn.disabled = true;
			retrainBtn.textContent = '재학습 중...';
			const result = await startRetraining();
			if (result.success) {
				retrainBtn.textContent = '재학습 시작됨';
				// 재학습 시작 후 상태 모니터링
				startStatusMonitoring();
			} else {
				retrainBtn.textContent = '재학습 실패';
				setTimeout(() => {
					retrainBtn.disabled = false;
					retrainBtn.textContent = '재학습 시작';
                    BUSY = false;
                }, 3000);
			}
            // 재학습이 시작되면 BUSY는 모니터링이 끝날 때 해제
		};
	}
	
	// 모델 재로드 버튼 이벤트
	const reloadBtn = document.getElementById('reloadModel');
	if (reloadBtn) {
		reloadBtn.onclick = async () => {
			reloadBtn.disabled = true;
			reloadBtn.textContent = '재로드 중...';
			const result = await reloadModel();
			if (result.success) {
				reloadBtn.textContent = '재로드 완료';
			} else {
				reloadBtn.textContent = '재로드 실패';
			}
			setTimeout(() => {
				reloadBtn.disabled = false;
				reloadBtn.textContent = '모델 재로드';
			}, 2000);
		};
	}

    // 학습 데이터 관리 버튼 이벤트
    const refreshBtn = document.getElementById('refreshData');
    const deleteBtn = document.getElementById('deleteAllData');
    if (refreshBtn) {
        refreshBtn.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            await updateTrainingStats();
        };
    }
    if (deleteBtn) {
        deleteBtn.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (BUSY) return;
            BUSY = true;
            deleteBtn.disabled = true;
            deleteBtn.textContent = '삭제 중...';
            if (retrainBtn) retrainBtn.disabled = true; // 오동작 방지
            const ok = await deleteAllTrainingData(); // 이제 영구만 삭제
            await updateTrainingStats();
            deleteBtn.textContent = ok ? '삭제 완료' : '삭제 실패';
            setTimeout(() => {
                deleteBtn.disabled = false;
                deleteBtn.textContent = '전체 삭제';
                if (retrainBtn) retrainBtn.disabled = false;
                BUSY = false;
            }, 1500);
        };
    }
}

function startStatusMonitoring() {
	const interval = setInterval(async () => {
		await updateTrainingStatus();
		await updateTrainingStats();
		
		const status = await getTrainingStatus();
		if (!status.is_training) {
			clearInterval(interval);
            BUSY = false;
		}
	}, 2000); // 2초마다 상태 확인
}

document.addEventListener('DOMContentLoaded', initPopup);


