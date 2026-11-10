/* O Falsificador e o Detetive — controle da apresentação.
   Teclas: → / PageDown = avançar · ← / PageUp = voltar · F = tela cheia
           1–4 = escolher no jogo · Espaço = ação do slide (pausar, animar) */
(() => {
  const M = window.MANIFEST;
  const $ = (s) => document.querySelector(s);
  const A = (p) => "assets/" + p;
  const fmt = (n) => n.toLocaleString("pt-BR");
  const shuffle = (a) => { a = a.slice(); for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; };
  const pick = (a) => a[Math.floor(Math.random() * a.length)];

  if (!M) {
    document.body.innerHTML = '<p style="padding:4rem;font-size:2rem">Imagens não encontradas. Rode <code>scripts/export_assets.py</code> primeiro.</p>';
    return;
  }

  // pré-carrega tudo para não haver atraso no telão
  const all = [...M.real, ...M.stages.flatMap((s) => s.fakes), ...M.timelapse.map((t) => t.img), ...M.morph.flat()];
  all.forEach((p) => { new Image().src = A(p); });

  const slides = [...document.querySelectorAll(".slide")];
  const handlers = {};
  let cur = 0, step = 0;

  // ---------------- título ----------------
  handlers.titulo = {
    enter() {
      const box = $("#title-faces");
      box.innerHTML = "";
      const last = M.stages[M.stages.length - 1].fakes;
      last.slice(0, 6).forEach((p, i) => {
        const img = new Image();
        img.src = A(p);
        img.alt = "Rosto gerado por IA";
        img.style.animationDelay = 0.3 + i * 0.15 + "s";
        box.appendChild(img);
      });
    },
  };

  // ---------------- história ----------------
  const story = {
    fakeImg: $("#story-fake img"), realImg: $("#story-real img"),
    stamp: $("#story-stamp"), caption: $("#story-caption"),
  };
  const S = M.stages;
  const storySteps = [
    { g: 1, d: 0, fake: S[0], real: 0, stamp: null,
      text: "O falsificador começa sem saber nada. Sua primeira tentativa de rosto é um borrão." },
    { g: 0, d: 1, fake: S[0], real: 1, stamp: ["FALSO!", "fake"],
      text: "O detetive compara com fotos reais. Fácil: falso!" },
    { g: 1, d: 1, fake: S[Math.min(1, S.length - 1)], real: 1, stamp: ["FALSO!", "fake"],
      text: "O falsificador descobre onde errou e tenta de novo. O detetive também treina e fica mais exigente." },
    { g: 1, d: 1, fake: S[S.length - 1], real: 1, stamp: ["REAL…?", "unsure"],
      text: "Depois de milhares de rodadas, o detetive já não tem certeza." },
    { g: 1, d: 1, fake: S[S.length - 1], real: 1, stamp: ["REAL…?", "unsure"], reveal: true, text: "" },
  ];
  let storyFakes = {}, storyReal;
  handlers.historia = {
    enter() { storyFakes = {}; storyReal = pick(M.real); },
    step(i) {
      const s = storySteps[i];
      $("#actor-g").classList.toggle("on", !!s.g);
      $("#actor-d").classList.toggle("on", !!s.d);
      storyFakes[s.fake.step] ??= pick(s.fake.fakes.slice(0, 4));
      story.fakeImg.src = A(storyFakes[s.fake.step]);
      story.realImg.src = A(storyReal);
      $("#story-real").classList.toggle("on", !!s.real);
      story.stamp.classList.remove("on");
      if (s.stamp) {
        story.stamp.textContent = s.stamp[0];
        story.stamp.className = "stamp " + s.stamp[1];
        setTimeout(() => story.stamp.classList.add("on"), 350);
      }
      story.caption.textContent = s.text;
      $("#story-reveal").classList.toggle("on", !!s.reveal);
    },
  };

  // ---------------- jogo ----------------
  const game = { round: 0, d: 0, g: 0, revealed: false, done: false, used: new Set() };
  let cardsEl;
  const slideJogo = slides.find((s) => s.dataset.name === "jogo");
  const gameOriginal = slideJogo.innerHTML;

  function newRound() {
    const st = M.stages[game.round];
    const reals = shuffle(M.real.filter((r) => !game.used.has(r))).slice(0, 3);
    reals.forEach((r) => game.used.add(r));
    const fake = pick(st.fakes.slice(0, 4)); // os mais convincentes daquele estágio
    const items = shuffle([...reals.map((src) => ({ src, fake: false })), { src: fake, fake: true }]);
    game.revealed = false;
    cardsEl.classList.remove("revealed");
    cardsEl.innerHTML = "";
    items.forEach((it, i) => {
      const c = document.createElement("div");
      c.className = "card " + (it.fake ? "is-fake" : "is-real");
      c.innerHTML = `<img src="${A(it.src)}" alt="Rosto ${i + 1}"><span class="num">${i + 1}</span><span class="tag">${it.fake ? "FALSO" : "real"}</span>`;
      c.onclick = () => choose(i);
      cardsEl.appendChild(c);
    });
    const n = M.stages.length;
    const hint = ["um começo difícil para ele", "ele está pegando o jeito", "agora fica sério", "treino completo"][Math.min(game.round, 3)];
    $("#round-info").innerHTML = `Rodada <b>${game.round + 1}</b> de ${n} · o falsificador treinou <b>${fmt(st.step)}</b> vezes — ${hint}`;
    $("#round-msg").innerHTML = 'Qual destes rostos é <strong>falso</strong>? <span class="hint">(clique ou tecle 1–4)</span>';
  }

  function bump(el) { el.classList.remove("bump"); void el.offsetWidth; el.classList.add("bump"); }

  function choose(i) {
    if (game.revealed || game.done) return;
    game.revealed = true;
    const cards = [...cardsEl.children];
    cardsEl.classList.add("revealed");
    if (i == null) {
      $("#round-msg").innerHTML = "Este era o falso.";
      return;
    }
    cards[i].classList.add("picked");
    if (cards[i].classList.contains("is-fake")) {
      game.d++; bump($("#score-d").parentElement);
      $("#round-msg").innerHTML = '<span class="good">Acertou!</span> Ponto para o detetive.';
    } else {
      game.g++; bump($("#score-g").parentElement);
      $("#round-msg").innerHTML = '<span class="bad">Enganado!</span> Ponto para o falsificador.';
    }
    $("#score-d").textContent = game.d;
    $("#score-g").textContent = game.g;
  }

  function finalScreen() {
    game.done = true;
    const n = M.stages.length;
    const verdict = game.g > 0
      ? "O falsificador conseguiu te enganar — e ele treinou só algumas horas num notebook."
      : "Você venceu desta vez! Mas repare como ficou mais difícil a cada rodada.";
    slideJogo.innerHTML = `<div class="final">
      <p class="big">Detetive ${game.d} × ${game.g} Falsificador</p>
      <p>${verdict}</p>
      <p>É exatamente assim que a GAN aprende: cada vez que o detetive acerta, o falsificador melhora.</p>
    </div>`;
  }

  handlers.jogo = {
    enter() {
      slideJogo.innerHTML = gameOriginal;
      cardsEl = $("#cards");
      Object.assign(game, { round: 0, d: 0, g: 0, revealed: false, done: false, used: new Set() });
      newRound();
    },
    next() { // retorna true se consumiu o avanço
      if (game.done) return false;
      if (!game.revealed) { choose(null); return true; }
      if (game.round < M.stages.length - 1) { game.round++; newRound(); }
      else finalScreen();
      return true;
    },
    key(k) { if (/^[1-4]$/.test(k)) { choose(+k - 1); return true; } },
  };

  // ---------------- timelapse ----------------
  const tl = { i: 0, timer: null };
  function tlShow(i) {
    const f = M.timelapse[i];
    $("#tl-img").src = A(f.img);
    $("#tl-step").textContent = fmt(f.step);
    $("#tl-bar").style.width = (100 * i / (M.timelapse.length - 1)) + "%";
  }
  function tlPlay() {
    clearInterval(tl.timer);
    if (tl.i >= M.timelapse.length - 1) tl.i = 0;
    tl.timer = setInterval(() => {
      tl.i++;
      tlShow(tl.i);
      if (tl.i >= M.timelapse.length - 1) { clearInterval(tl.timer); tl.timer = null; }
    }, 420);
  }
  handlers.timelapse = {
    enter() { tl.i = 0; tlShow(0); setTimeout(tlPlay, 700); },
    leave() { clearInterval(tl.timer); tl.timer = null; },
    space() { if (tl.timer) { clearInterval(tl.timer); tl.timer = null; } else tlPlay(); },
  };

  // ---------------- morph ----------------
  const mo = { pair: 0, timer: null, dir: 1 };
  const slider = $("#morph-slider");
  function moShow() { $("#morph-img").src = A(M.morph[mo.pair][+slider.value]); }
  slider.max = M.morph[0].length - 1;
  slider.oninput = moShow;
  function moToggle() {
    if (mo.timer) { clearInterval(mo.timer); mo.timer = null; return; }
    mo.timer = setInterval(() => {
      let v = +slider.value + mo.dir;
      if (v > +slider.max || v < 0) { mo.dir *= -1; v += 2 * mo.dir; }
      slider.value = v; moShow();
    }, 70);
  }
  handlers.morph = {
    enter() { slider.value = 0; mo.dir = 1; moShow(); },
    leave() { clearInterval(mo.timer); mo.timer = null; },
    space: moToggle,
    key(k) { if (k.toLowerCase() === "n") { mo.pair = (mo.pair + 1) % M.morph.length; moShow(); return true; } },
  };

  // ---------------- mundo real ----------------
  handlers.mundo = { step(i) { $("#end-reveal").classList.toggle("on", i >= 1); } };

  // ---------------- navegação ----------------
  const prog = $("#progress");
  slides.forEach(() => prog.appendChild(document.createElement("span")));

  function show(n, atEnd = false) {
    const h0 = handlers[slides[cur].dataset.name];
    h0?.leave?.();
    slides[cur].classList.remove("active");
    cur = Math.max(0, Math.min(slides.length - 1, n));
    const el = slides[cur], h = handlers[el.dataset.name];
    el.classList.add("active");
    const steps = +(el.dataset.steps || 1);
    step = atEnd ? steps - 1 : 0;
    h?.enter?.();
    h?.step?.(step);
    [...prog.children].forEach((d, i) => d.classList.toggle("on", i === cur));
  }

  function next() {
    const el = slides[cur], h = handlers[el.dataset.name];
    if (h?.next?.()) return;
    const steps = +(el.dataset.steps || 1);
    if (step < steps - 1) { step++; h?.step?.(step); return; }
    if (cur < slides.length - 1) show(cur + 1);
  }
  function prev() {
    const el = slides[cur], h = handlers[el.dataset.name];
    if (step > 0) { step--; h?.step?.(step); return; }
    if (cur > 0) show(cur - 1, true);
  }

  document.addEventListener("keydown", (e) => {
    const h = handlers[slides[cur].dataset.name];
    if (e.key === "ArrowRight" || e.key === "PageDown") { e.preventDefault(); next(); }
    else if (e.key === "ArrowLeft" || e.key === "PageUp") { e.preventDefault(); prev(); }
    else if (e.key === " ") { e.preventDefault(); h?.space ? h.space() : next(); }
    else if (e.key.toLowerCase() === "f") {
      document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen();
    } else h?.key?.(e.key);
  });

  show(0);
})();
