const DEF = { enabled: true, minSeverityToHide: 2, action: 'hide', showBadge: true, trainingMode: false, serverUrl: 'http://127.0.0.1:8000', rules: [], masks: [] };
function getValues(){ return new Promise(r => chrome.storage.local.get(DEF, r)); }
function setValues(v){ return new Promise(r => chrome.storage.local.set(v, r)); }

async function getTrainingStats() {
	const { serverUrl } = await getValues();
	const base = getTrainingServerBase(serverUrl);
	try {
		// 영구 데이터만 (사용자가 클릭해서 수집한 데이터)
		const response = await fetch(`${base}/training-data/stats`);
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
    const base = getTrainingServerBase(serverUrl);
    const res = await fetch(`${base}/training-data/files`);
    if (!res.ok) return { files: [] };
    return await res.json();
}

async function getTrainingFileContent(filename) {
    const { serverUrl } = await getValues();
    const base = getTrainingServerBase(serverUrl);
    const res = await fetch(`${base}/training-data/files/${filename}`);
    if (!res.ok) return { data: [] };
    return await res.json();
}

async function deleteTrainingFile(filename) {
    const { serverUrl } = await getValues();
    const base = getTrainingServerBase(serverUrl);
    const res = await fetch(`${base}/training-data/files/${filename}`, { method: 'DELETE' });
    return res.ok;
}

async function deleteTrainingLine(filename, lineNumber) {
    const { serverUrl } = await getValues();
    const base = getTrainingServerBase(serverUrl);
    const res = await fetch(`${base}/training-data/files/${filename}/lines/${lineNumber}`, { method: 'DELETE' });
    return res.ok;
}

async function deleteAllTrainingData() {
    const { serverUrl } = await getValues();
    const base = getTrainingServerBase(serverUrl);
    const res = await fetch(`${base}/training-data/all`, { method: 'DELETE' });
    return res.ok;
}

async function deleteTempOnly() {
    const { serverUrl } = await getValues();
    const base = getTrainingServerBase(serverUrl);
    const res = await fetch(`${base}/training-data/temp`, { method: 'DELETE' });
    return res.ok;
}

async function startRetraining() {
	const { serverUrl } = await getValues();
	// 외부 서버(포트 3000)는 재학습 비활성화
	try {
		const u = new URL(serverUrl || '');
		if (u.port === '3000' || /223\.194\.46\.69/.test(u.hostname)) {
			return { success: false, message: '외부 서버에서는 재학습을 지원하지 않습니다.' };
		}
	} catch {
		if (/:3000$/.test(String(serverUrl||''))) {
			return { success: false, message: '외부 서버에서는 재학습을 지원하지 않습니다.' };
		}
	}
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
	// 외부 서버(포트 3000)는 모델 재로드 비활성화
	try {
		const u = new URL(serverUrl || '');
		if (u.port === '3000' || /223\.194\.46\.69/.test(u.hostname)) {
			return { success: false, message: '외부 서버에서는 모델 재로드를 지원하지 않습니다.' };
		}
	} catch {
		if (/:3000$/.test(String(serverUrl||''))) {
			return { success: false, message: '외부 서버에서는 모델 재로드를 지원하지 않습니다.' };
		}
	}
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
	// 외부 서버(포트 3000)는 상태 조회 비활성화
	try {
		const u = new URL(serverUrl || '');
		if (u.port === '3000' || /223\.194\.46\.69/.test(u.hostname)) {
			return { is_training: false, progress: 0, message: '외부 서버 모드', error: null };
		}
	} catch {
		if (/:3000$/.test(String(serverUrl||''))) {
			return { is_training: false, progress: 0, message: '외부 서버 모드', error: null };
		}
	}
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
const SELECTED_FILES = new Set();

function getTrainingServerBase(serverUrl) {
	try {
		const u = new URL(serverUrl || '');
		if (u.port === '3000' || /223\.194\.46\.69/.test(u.hostname)) {
			return 'http://127.0.0.1:8000';
		}
	} catch {
		if (/:3000$/.test(String(serverUrl||''))) {
			return 'http://127.0.0.1:8000';
		}
	}
	return (serverUrl || '').replace(/\/$/, '');
}
async function initPopup() {
	const v = await getValues();
	const min = document.getElementById('min');
	const act = document.getElementById('act');
	const badge = document.getElementById('badge');
	const serverUrlInput = document.getElementById('serverUrlInput');
	const applyServerUrl = document.getElementById('applyServerUrl');
	const testServerUrl = document.getElementById('testServerUrl');
	const setLocalServer = document.getElementById('setLocalServer');
	const setExternalServer = document.getElementById('setExternalServer');
	const useExternalServer = document.getElementById('useExternalServer');
	const count = document.getElementById('count');
	const trainingMode = document.getElementById('trainingMode');
    const enabled = document.getElementById('enabled');
	
	min.value = String(v.minSeverityToHide);
	act.value = v.action;
	badge.checked = !!v.showBadge;
	if (serverUrlInput) serverUrlInput.value = v.serverUrl || 'http://127.0.0.1:8000';
	// 버튼 활성 상태 표시
	const updateServerButtons = (url) => {
		if (!setLocalServer || !setExternalServer) return;
		const u = (url || '').trim().replace(/\/$/, '');
		const isLocal = /^https?:\/\/(127\.0\.0\.1|localhost):8000$/.test(u);
		const isExternal = /^https?:\/\/223\.194\.46\.69:3000$/.test(u);
		const setState = (btn, active) => {
			if (!btn) return;
			btn.classList.toggle('btn-primary', !!active);
			btn.classList.toggle('btn-secondary', !active);
		};
		setState(setLocalServer, isLocal);
		setState(setExternalServer, isExternal && !isLocal);
	};
	// 외부 서버 모드 UI 적용/해제
	const setExternalModeUI = (isExternal) => {
		const retrainBtn0 = document.getElementById('retrainModel');
		const reloadBtn0 = document.getElementById('reloadModel');
		const reloadRow0 = reloadBtn0 && reloadBtn0.closest('.row');
		const retrainRow0 = retrainBtn0 && retrainBtn0.closest('.row');
		const progressDiv0 = document.getElementById('trainingProgress');
		if (isExternal) {
			// 모델 관리 행 숨김
			if (reloadRow0) reloadRow0.style.display = 'none';
			// 재학습 행도 숨김
			if (retrainRow0) retrainRow0.style.display = 'none';
			// 진행 표시 숨김
			if (progressDiv0) progressDiv0.style.display = 'none';
			// 수집 모드 토글은 외부 서버에서도 사용 가능
			if (trainingMode) { trainingMode.disabled = false; }
		} else {
			// 모델 관리 행 표시
			if (reloadRow0) reloadRow0.style.display = '';
			// 재학습 행 표시
			if (retrainRow0) retrainRow0.style.display = '';
			// 재학습 버튼 활성화 및 텍스트 복구
			if (retrainBtn0) { retrainBtn0.disabled = false; retrainBtn0.textContent = '재학습 시작'; }
			// 수집 모드 토글 활성화
			if (trainingMode) { trainingMode.disabled = false; }
		}
	};
	const currentUrl = v.serverUrl || 'http://127.0.0.1:8000';
	updateServerButtons(currentUrl);
	// 외부 서버 토글 상태 설정
	const isExternalNow = /^https?:\/\/223\.194\.46\.69:3000$/.test((currentUrl || '').trim().replace(/\/$/, ''));
	if (useExternalServer) useExternalServer.checked = isExternalNow;
	setExternalModeUI(isExternalNow);
	trainingMode.checked = !!v.trainingMode;
    if (enabled) enabled.checked = v.enabled !== false;
	
	chrome.action.getBadgeText({}, t => count.textContent = t || '0');
	
	min.onchange = () => setValues({ minSeverityToHide: Number(min.value) });
	act.onchange = () => setValues({ action: act.value });
	badge.onchange = () => setValues({ showBadge: badge.checked });
	if (applyServerUrl && serverUrlInput) {
		applyServerUrl.onclick = async (e) => {
			e.preventDefault(); e.stopPropagation();
			const url = (serverUrlInput.value || '').trim();
			if (!url) return;
			const prev = applyServerUrl.textContent;
			applyServerUrl.textContent = '저장 중...';
			applyServerUrl.disabled = true;
			await setValues({ serverUrl: url });
			setTimeout(() => {
				applyServerUrl.textContent = '적용됨';
				setTimeout(() => {
					applyServerUrl.textContent = prev || '적용';
					applyServerUrl.disabled = false;
				}, 800);
			}, 150);
		};
	}
	if (testServerUrl && serverUrlInput) {
		testServerUrl.onclick = async (e) => {
			e.preventDefault(); e.stopPropagation();
			const url = (serverUrlInput.value || '').trim().replace(/\/$/, '');
			if (!url) return;
			const prev = testServerUrl.textContent;
			testServerUrl.textContent = '점검 중...';
			testServerUrl.disabled = true;
			try {
				const res = await fetch(`${url}/health`);
				testServerUrl.textContent = res.ok ? '연결 OK' : `HTTP ${res.status}`;
			} catch (err) {
				testServerUrl.textContent = '연결 실패';
			}
			setTimeout(() => {
				testServerUrl.textContent = prev || '점검';
				testServerUrl.disabled = false;
			}, 1200);
		};
	}
	if (setLocalServer && serverUrlInput) {
		setLocalServer.onclick = async (e) => {
			e.preventDefault(); e.stopPropagation();
			const target = 'http://127.0.0.1:8000';
			serverUrlInput.value = target;
			await setValues({ serverUrl: target });
			updateServerButtons(target);
			setExternalModeUI(false);
			try { testServerUrl?.click(); } catch {}
		};
	}
	if (setExternalServer && serverUrlInput) {
		setExternalServer.onclick = async (e) => {
			e.preventDefault(); e.stopPropagation();
			const target = 'http://223.194.46.69:3000';
			serverUrlInput.value = target;
			await setValues({ serverUrl: target });
			updateServerButtons(target);
			setExternalModeUI(true);
			try { testServerUrl?.click(); } catch {}
		};
	}
	if (useExternalServer && serverUrlInput) {
		useExternalServer.onchange = async (e) => {
			e.preventDefault(); e.stopPropagation();
			const target = useExternalServer.checked ? 'http://223.194.46.69:3000' : 'http://127.0.0.1:8000';
			serverUrlInput.value = target;
			await setValues({ serverUrl: target });
			updateServerButtons(target);
			setExternalModeUI(useExternalServer.checked);
			try { testServerUrl?.click(); } catch {}
		};
	}
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
            await renderTrainingDataFiles();
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
            await renderTrainingDataFiles();
            deleteBtn.textContent = ok ? '삭제 완료' : '삭제 실패';
            setTimeout(() => {
                deleteBtn.disabled = false;
                deleteBtn.textContent = '전체 삭제';
                if (retrainBtn) retrainBtn.disabled = false;
                BUSY = false;
            }, 1500);
        };
    }
    
    // 데이터 보기 토글 버튼
    const toggleDataBtn = document.getElementById('toggleDataView');
    const dataViewSection = document.getElementById('dataViewSection');
    if (toggleDataBtn && dataViewSection) {
        toggleDataBtn.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            if (dataViewSection.style.display === 'none') {
                dataViewSection.style.display = 'block';
                toggleDataBtn.textContent = '📋 수집된 데이터 숨기기';
                // 처음 열 때 데이터 로드
                await renderTrainingDataFiles();
            } else {
                dataViewSection.style.display = 'none';
                toggleDataBtn.textContent = '📋 수집된 데이터 보기';
            }
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

async function renderRules() {
  const v = await getValues();
  const list = document.getElementById('ruleList');
  if (!list) return;
  const rules = Array.isArray(v.rules) ? v.rules : [];
  if (rules.length === 0) {
    list.innerHTML = '<div class="help">등록된 룰이 없습니다.</div>';
    return;
  }
  list.innerHTML = rules.map((r, idx) =>
    `<div class="kv"><span class="dot"></span> <strong>${r.term}</strong> → 최소 ${r.min}
     <button data-idx="${idx}" class="btn btn-small" style="margin-left:8px;">삭제</button></div>`
  ).join('');
  list.querySelectorAll('button[data-idx]').forEach(btn => {
    btn.onclick = async () => {
      const i = Number(btn.dataset.idx);
      const curr = await getValues();
      const arr = Array.isArray(curr.rules) ? curr.rules.slice() : [];
      if (i >= 0 && i < arr.length) arr.splice(i, 1);
      await setValues({ rules: arr });
      renderRules();
    };
  });
}

async function renderMasks() {
  const v = await getValues();
  const list = document.getElementById('maskList');
  if (!list) return;
  const masks = Array.isArray(v.masks) ? v.masks : [];
  if (masks.length === 0) {
    list.innerHTML = '<div class="help">등록된 마스크가 없습니다.</div>';
    return;
  }
  list.innerHTML = masks.map((m, idx) =>
    `<div class="kv"><span class="dot"></span> <strong>${m}</strong>
     <button data-idx="${idx}" class="btn btn-small" style="margin-left:8px;">삭제</button></div>`
  ).join('');
  list.querySelectorAll('button[data-idx]').forEach(btn => {
    btn.onclick = async () => {
      const i = Number(btn.dataset.idx);
      const curr = await getValues();
      const arr = Array.isArray(curr.masks) ? curr.masks.slice() : [];
      if (i >= 0 && i < arr.length) arr.splice(i, 1);
      await setValues({ masks: arr });
      renderMasks();
    };
  });
}

async function installRuleMaskHandlers() {
  const addRule = document.getElementById('addRule');
  const ruleTerm = document.getElementById('ruleTerm');
  const ruleMin = document.getElementById('ruleMin');

  if (addRule && ruleTerm && ruleMin) {
    addRule.onclick = async (e) => {
      e.preventDefault(); e.stopPropagation();
      const term = (ruleTerm.value || '').trim();
      const min = Number(ruleMin.value || 0);
      if (!term) return;
      const curr = await getValues();
      const rules = Array.isArray(curr.rules) ? curr.rules.slice() : [];
      // 중복 용어는 최신 설정으로 갱신
      const idx = rules.findIndex(r => r.term === term);
      if (idx >= 0) rules[idx] = { term, min }; else rules.push({ term, min });
      await setValues({ rules });
      ruleTerm.value = '';
      renderRules();
    };
  }

  const addMask = document.getElementById('addMask');
  const maskTerm = document.getElementById('maskTerm');
  if (addMask && maskTerm) {
    addMask.onclick = async (e) => {
      e.preventDefault(); e.stopPropagation();
      const term = (maskTerm.value || '').trim();
      if (!term) return;
      const curr = await getValues();
      const masks = Array.isArray(curr.masks) ? curr.masks.slice() : [];
      if (!masks.includes(term)) masks.push(term);
      await setValues({ masks });
      maskTerm.value = '';
      renderMasks();
    };
  }

  // 최초 렌더
  await renderRules();
  await renderMasks();
}

async function renderTrainingDataFiles() {
    const dataFilesDiv = document.getElementById('dataFiles');
    if (!dataFilesDiv) return;
    
    const result = await getTrainingFiles();
    const files = result.files || [];
    // 선택 상태에서 사라진 파일 정리
    for (const fn of Array.from(SELECTED_FILES)) {
        if (!files.find(f => f.filename === fn)) SELECTED_FILES.delete(fn);
    }
    
    if (files.length === 0) {
        dataFilesDiv.innerHTML = '<div class="help">저장된 학습 데이터가 없습니다.</div>';
        const selectAll = document.getElementById('selectAllFiles');
        if (selectAll) selectAll.checked = false;
        return;
    }
    
    let html = '<div style="max-height: 300px; overflow-y: auto;">';
    for (const file of files) {
        html += `
            <div class="data-file-item" style="margin-bottom: 12px; padding: 8px; background: rgba(255,255,255,0.05); border-radius: 4px;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <div style="display:flex; align-items:center; gap:8px;">
                        <input type="checkbox" class="file-select" data-filename="${file.filename}">
                        <div style="font-weight: 600;">${file.filename}</div>
                    </div>
                    <button class="btn btn-small btn-danger delete-file-btn" data-filename="${file.filename}">파일 삭제</button>
                </div>
                <div style="font-size: 12px; color: #aaa;">
                    ${file.count}개 항목 | ${(file.size / 1024).toFixed(1)}KB | ${new Date(file.date).toLocaleDateString()}
                </div>
                <div style="margin-top: 8px;">
                    <button class="btn btn-small view-content-btn" data-filename="${file.filename}">내용 보기</button>
                </div>
                <div class="file-content" id="content-${file.filename}" style="display: none; margin-top: 8px;"></div>
            </div>
        `;
    }
    html += '</div>';
    
    dataFilesDiv.innerHTML = html;
    
    // 체크박스 바인딩
    const checkboxes = dataFilesDiv.querySelectorAll('.file-select');
    checkboxes.forEach(cb => {
        const fn = cb.getAttribute('data-filename');
        if (SELECTED_FILES.has(fn)) cb.checked = true;
        cb.addEventListener('change', () => {
            if (cb.checked) SELECTED_FILES.add(fn); else SELECTED_FILES.delete(fn);
            const selectAll = document.getElementById('selectAllFiles');
            if (selectAll) {
                const total = checkboxes.length;
                const selected = Array.from(checkboxes).filter(x => x.checked).length;
                selectAll.checked = total > 0 && selected === total;
            }
        });
    });
    
    const selectAll = document.getElementById('selectAllFiles');
    if (selectAll) {
        // 초기 상태 동기화
        const total = checkboxes.length;
        const selected = Array.from(checkboxes).filter(x => x.checked).length;
        selectAll.checked = total > 0 && selected === total;
        // 변경 핸들러
        selectAll.onchange = () => {
            const checked = !!selectAll.checked;
            checkboxes.forEach(cb => {
                cb.checked = checked;
                const fn = cb.getAttribute('data-filename');
                if (checked) SELECTED_FILES.add(fn); else SELECTED_FILES.delete(fn);
            });
        };
    }
    
    const deleteSelectedBtn = document.getElementById('deleteSelectedFiles');
    if (deleteSelectedBtn) {
        deleteSelectedBtn.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const targets = Array.from(SELECTED_FILES);
            if (targets.length === 0) {
                alert('선택된 파일이 없습니다.');
                return;
            }
            if (!confirm(`${targets.length}개 파일을 삭제하시겠습니까?`)) return;
            deleteSelectedBtn.disabled = true;
            const originalText = deleteSelectedBtn.textContent;
            let done = 0;
            for (const fn of targets) {
                deleteSelectedBtn.textContent = `삭제 중... (${++done}/${targets.length})`;
                try { await deleteTrainingFile(fn); } catch {}
            }
            SELECTED_FILES.clear();
            const sa = document.getElementById('selectAllFiles');
            if (sa) sa.checked = false;
            await updateTrainingStats();
            await renderTrainingDataFiles();
            deleteSelectedBtn.disabled = false;
            deleteSelectedBtn.textContent = originalText || '선택 삭제';
        };
    }
    
    // 파일 삭제 버튼
    dataFilesDiv.querySelectorAll('.delete-file-btn').forEach(btn => {
        btn.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const filename = btn.dataset.filename;
            if (!confirm(`${filename}을(를) 삭제하시겠습니까?`)) return;
            
            btn.disabled = true;
            btn.textContent = '삭제 중...';
            const ok = await deleteTrainingFile(filename);
            if (ok) {
                await updateTrainingStats();
                await renderTrainingDataFiles();
            } else {
                alert('파일 삭제 실패');
                btn.disabled = false;
                btn.textContent = '파일 삭제';
            }
        };
    });
    
    // 내용 보기 버튼
    dataFilesDiv.querySelectorAll('.view-content-btn').forEach(btn => {
        btn.onclick = async (e) => {
            e.preventDefault();
            e.stopPropagation();
            const filename = btn.dataset.filename;
            const contentDiv = document.getElementById(`content-${filename}`);
            
            if (contentDiv.style.display === 'none') {
                // 내용 로드 및 표시
                btn.textContent = '로딩...';
                btn.disabled = true;
                
                const result = await getTrainingFileContent(filename);
                const data = result.data || [];
                
                if (data.length === 0) {
                    contentDiv.innerHTML = '<div class="help">데이터가 없습니다.</div>';
                } else {
                    let contentHtml = '<div style="max-height: 200px; overflow-y: auto; font-size: 11px;">';
                    for (const item of data) {
                        const labelName = item.label === 0 ? '정상' : item.label === 1 ? '약간 악성' : '악성';
                        const labelColor = item.label === 0 ? '#5bc0de' : item.label === 1 ? '#f0ad4e' : '#d9534f';
                        contentHtml += `
                            <div style="padding: 6px; margin-bottom: 4px; background: rgba(0,0,0,0.2); border-radius: 3px; display: flex; justify-content: space-between; align-items: start;">
                                <div style="flex: 1;">
                                    <span style="color: ${labelColor}; font-weight: 600;">[${labelName}]</span>
                                    <span style="margin-left: 8px;">${item.text.substring(0, 80)}${item.text.length > 80 ? '...' : ''}</span>
                                </div>
                                <button class="btn btn-small delete-line-btn" data-filename="${filename}" data-line="${item.line_number}" style="margin-left: 8px;">삭제</button>
                            </div>
                        `;
                    }
                    contentHtml += '</div>';
                    contentDiv.innerHTML = contentHtml;
                    
                    // 라인 삭제 버튼
                    contentDiv.querySelectorAll('.delete-line-btn').forEach(lineBtn => {
                        lineBtn.onclick = async (e) => {
                            e.preventDefault();
                            e.stopPropagation();
                            const fn = lineBtn.dataset.filename;
                            const ln = parseInt(lineBtn.dataset.line);
                            
                            lineBtn.disabled = true;
                            lineBtn.textContent = '삭제중';
                            const ok = await deleteTrainingLine(fn, ln);
                            if (ok) {
                                await updateTrainingStats();
                                // 내용 다시 로드
                                btn.click();
                                setTimeout(() => btn.click(), 100);
                            } else {
                                alert('삭제 실패');
                                lineBtn.disabled = false;
                                lineBtn.textContent = '삭제';
                            }
                        };
                    });
                }
                
                contentDiv.style.display = 'block';
                btn.textContent = '내용 숨기기';
                btn.disabled = false;
            } else {
                // 숨기기
                contentDiv.style.display = 'none';
                btn.textContent = '내용 보기';
            }
        };
    });
}

// 기존 initPopup 끝부분에서 호출(또는 DOMContentLoaded 시점)
document.addEventListener('DOMContentLoaded', () => {
  // 기존 initPopup가 이미 바인딩되어 있으면 그 이후에 호출해도 무방
  try { installRuleMaskHandlers(); } catch {}
});
