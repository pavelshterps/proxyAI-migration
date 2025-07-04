<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8" />
  <title>proxyAI Demo UI</title>
  <style>
    body { font-family: sans-serif; margin: 2rem; }
    label, button { margin: .5rem 0; display: block; }
    #transcriptContainer { margin-top: 1rem; }
    .segment { border-bottom: 1px solid #ccc; padding: .5rem 0; }
    .segment input { width: 180px; margin-left: .5rem; }
    .segment button { margin-left: .5rem; }
    .time { color: #666; margin-right: .5rem; }
  </style>
</head>
<body>
  <h1>proxyAI Demo UI</h1>

  <label>
    Ваш API-Key:
    <input type="text" id="apiKey" placeholder="X-API-Key" />
  </label>

  <label>
    Аудио-файл:
    <input type="file" id="audioFile" accept="audio/*" />
  </label>
  <button id="uploadBtn">Загрузить и начать</button>

  <div id="status" style="margin-top:1rem;"></div>
  <button id="fetchResultsBtn" disabled>Получить результаты</button>

  <h2>Плеер</h2>
  <audio id="audio" controls style="width:100%"></audio>

  <h2>Транскрипция / Диаризация</h2>
  <div id="transcriptContainer"></div>

  <button id="saveLabelsBtn" disabled>Сохранить метки спикеров</button>

<script>
(async function(){
  const apiKeyEl   = document.getElementById('apiKey');
  const fileEl     = document.getElementById('audioFile');
  const uploadBtn  = document.getElementById('uploadBtn');
  const statusEl   = document.getElementById('status');
  const fetchBtn   = document.getElementById('fetchResultsBtn');
  const audioEl    = document.getElementById('audio');
  const container  = document.getElementById('transcriptContainer');
  const saveBtn    = document.getElementById('saveLabelsBtn');

  let uploadId, fileBlob, mapping = {};

  uploadBtn.onclick = async () => {
    const key  = apiKeyEl.value.trim();
    if (!key) { alert("Введите API-Key"); return; }
    const file = fileEl.files[0];
    if (!file){ alert("Выберите аудио файл"); return; }
    fileBlob = file;

    statusEl.textContent = "Загружаем...";
    const fd = new FormData();
    fd.append("file", file, file.name);

    const res = await fetch("/upload/", {
      method: "POST",
      headers: { "X-API-Key": key },
      body: fd
    });
    if (!res.ok) { alert("Upload error: "+res.status); return; }
    const { upload_id } = await res.json();
    uploadId = upload_id;
    statusEl.textContent = `upload_id=${uploadId}. Ожидаем...`;

    const interval = setInterval(async ()=>{
      const st = await (await fetch(`/status/${uploadId}`, {
        headers:{ "X-API-Key": key }
      })).json();
      statusEl.textContent = `Статус: ${st.status}, прогресс: ${st.progress}`;
      if (st.status === "done" || st.progress === "100%") {
        clearInterval(interval);
        fetchBtn.disabled = false;
      }
    }, 2000);
  };

  fetchBtn.onclick = async () => {
    const key = apiKeyEl.value.trim();
    statusEl.textContent = "Получаем результаты...";
    const res = await fetch(`/results/${uploadId}`, {
      headers:{ "X-API-Key": key }
    });
    if (!res.ok) { alert("Error fetching results"); return; }
    // Единожды парсим JSON
    const payload = await res.json();
    // поддерживаем оба поля
    const list = payload.results ?? payload.transcript;
    statusEl.textContent = "Результаты получены";
    audioEl.src = URL.createObjectURL(fileBlob);
    renderTranscript(list);
    saveBtn.disabled = false;
  };

  function renderTranscript(list){
    container.innerHTML = "";
    mapping = {};
    list.forEach(seg => mapping[seg.speaker] = seg.speaker);
    list.forEach(seg => {
      const div = document.createElement("div");
      div.className = "segment";

      const t = document.createElement("span");
      t.className = "time";
      t.textContent = seg.time || "--:--:--";

      const btn = document.createElement("button");
      btn.textContent = "▶️";
      btn.onclick = () => {
        audioEl.currentTime = seg.start;
        audioEl.play();
        setTimeout(()=> audioEl.pause(), (seg.end - seg.start)*1000 + 200);
      };

      const inp = document.createElement("input");
      inp.value = seg.speaker;
      inp.onchange = () => {
        const old = seg.speaker, neu = inp.value.trim();
        if (neu && neu.length <= 50) {
          mapping[old] = neu;
          document.querySelectorAll('.segment input').forEach(i => {
            if (i.value === old) i.value = neu;
          });
        } else {
          alert("Имя 1–50 символов");
          inp.value = mapping[old];
        }
      };

      const txt = document.createElement("span");
      txt.textContent = seg.text;

      div.append(t, btn, inp, txt);
      container.append(div);
    });
  }

  saveBtn.onclick = async () => {
    const key = apiKeyEl.value.trim();
    const payload = {};
    Object.entries(mapping).forEach(([o,n]) => {
      if (o !== n) payload[o] = n;
    });
    if (!Object.keys(payload).length) { alert("Нет изменений"); return; }

    const res = await fetch(`/labels/${uploadId}`, {
      method: "POST",
      headers:{
        "Content-Type":"application/json",
        "X-API-Key": key
      },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      alert("Ошибка: "+(await res.json()).detail);
      return;
    }
    alert("Метки сохранены");
  };
})();
</script>
</body>
</html>