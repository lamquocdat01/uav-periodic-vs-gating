# THEORY_28 — When is periodic detection enough for UAV traffic monitoring? (v1, 2026-09-27)

*Self-contained theory for topic #28 (P1 of the plan). Every statement lists its assumptions, the statement, a complete proof, a numerical check (`code/theory_checks.py`, seed = 42, output `results/p1/theory_checks.json`) and how it will be tested on data. Where a planned statement turned out to be false or only true in part of the parameter space, the counterexample is given and the statement is corrected (§4.3, §5.4, §5.5). The only external work this note relates to is [Deep3], cited as a related derivation of a gain–cost threshold for frame-level gating; nothing below depends on it.*

**Status of the numerical checks (theory_checks.json):** 18 checks (v1.1 adds Remark G2′); 17 PASS. One check fails the literal per-comparison 3σ rule (PASS family-wise): in the 315-comparison Monte Carlo family for Corollary 1.3 one comparison lies at 3.002σ (expected number of >3σ exceedances if the formula is exact: 0.85; aggregate χ²(315) p = 0.9999). The corollary is an exact consequence of Lemma 1, so this is multiple-testing noise, but it is recorded as FAIL under the pre-set rule rather than re-seeded.

---

## 1. Notation and standing assumptions (B1)

| Symbol | Meaning |
|---|---|
| $t \in \mathbb{Z}$ | frame index of the capture stream, capture rate $F$ frames/s |
| event $e$ | one object-level event: onset (birth) frame $b_e$, visible duration $D_e \ge 1$ frames (the event occupies frames $b_e,\dots,b_e+D_e-1$) |
| $S \ge 1$ | stride of a periodic detector schedule (real-valued allowed, §2) |
| $\varphi$ | phase of the schedule, uniform on $[0,S)$ (the "uniform-phase" or randomised schedule) |
| $N_e$ | number of detector runs ("looks") that fall on frames of event $e$ |
| $r$ | per-look recall: probability that one detector run on a frame of the event detects it |
| $a$ | activation = long-run fraction of frames on which the detector runs (periodic: $a = 1/S$) |
| $f_{det}$ | detector rate in Hz: $f_{det} = aF = F/S$ |
| $L_e$ | onset latency: frames from $b_e$ to the first *successful* look; $L_e = \infty$ if none |
| $\mathrm{miss}$ | probability that an event receives no successful look in its KPI window |

Standing assumptions, invoked by name:

- **A-ind** (independent looks): conditional on which frames are looked at, each look on a frame of $e$ succeeds independently with probability $r$.
- **A-phase**: the schedule phase is uniform and independent of the events (equivalently, a deterministic schedule and event onsets that are uniformly distributed modulo $S$ — true on average under stationarity).
- **A-stat** (for §2.3 only): onsets and durations are independent of the observation window (the recording start/end is not triggered by the events).

**Periodic schedule with real stride.** For real $S \ge 1$ the schedule looks at frames $\lceil \varphi + kS \rceil$, $k \in \mathbb{Z}$, $\varphi \sim U[0,S)$. For integer $S$ this is the usual "every $S$-th frame" with a uniform integer phase. Decimation by $k$ followed by a detector every $R$ retained frames is the periodic schedule with $S = kR$ on the original frames.

---

## 2. Look-count law and censoring (B2)

### Lemma 1 (look-count law)
*Assumptions:* A-phase; periodic schedule with (real) stride $S \ge 1$; event of $D$ consecutive frames. Write $D/S = n + f$ with $n = \lfloor D/S \rfloor$, $f \in [0,1)$.

*Statement.* (i) $N = n + \mathrm{Bernoulli}(f)$. (ii) Under A-ind,
$$\mathrm{miss}_r(D,S) = \mathbb{E}[(1-r)^N] = (1-r)^{n}\,(1 - f + f(1-r)) = (1-r)^{n}(1 - fr).$$
(iii) For $r = 1$ (with $0^0 = 1$): $\mathrm{miss}_1(D,S) = (1 - D/S)_+$.

*Proof.* Look $k$ lands on the event iff $b \le \lceil \varphi + kS\rceil \le b + D - 1$. For integers $m$, $\lceil x\rceil \le m \iff x \le m$ and $\lceil x \rceil \ge m \iff x > m-1$; hence the condition is $\varphi + kS \in (b-1,\; b-1+D]$, a half-open interval of length $D$. So $N$ is the number of points of the lattice $\varphi + S\mathbb{Z}$ in a half-open interval of length $D = nS + fS$. Cut the interval into $n$ consecutive half-open pieces of length $S$ followed by a remainder of length $fS$. Each piece of length $S$ contains exactly one lattice point. The remainder contains at most one point, and because $\varphi$ is uniform on $[0,S)$ the position of the lattice modulo $S$ is uniform, so it contains a point with probability $fS/S = f$. This gives (i). Under A-ind, $P(\text{miss}\mid N) = (1-r)^N$; averaging over (i) gives $(1-f)(1-r)^n + f(1-r)^{n+1}$, which is (ii). For $r=1$, $(1-r)^n = 0$ unless $n = 0$, in which case $\mathrm{miss} = 1-f = 1 - D/S$; this is (iii). ∎

### Lemma 2 (monotonicity)
*Statement.* For fixed $S$ and $r\in(0,1]$, $D \mapsto \mathrm{miss}_r(D,S)$ is continuous (in real $D$), piecewise linear and non-increasing.

*Proof.* On $[nS,(n+1)S)$, $\mathrm{miss}_r = (1-r)^n\bigl(1 - r(D-nS)/S\bigr)$ is linear with slope $-(1-r)^n r/S \le 0$. As $D \uparrow (n+1)S$ it tends to $(1-r)^n(1-r) = (1-r)^{n+1}$, the value at $D=(n+1)S$. ∎

### Corollary 1 (right-censoring; justification of the birth window)
*Assumptions:* a sequence with frames $1,\dots,L_s$; an event with onset $b$ and true duration $D$ is observed with $D_{obs} = \min(D, L_s - b + 1)$ (right-censored if $D > L_s-b+1$). Birth window of width $W$: keep events with $b \le L_s - W$ that are not left-censored ($b > 1$).

*Statement.* (i) For $r = 1$ and every $S \le W$: $\mathrm{miss}_1(D_{obs}, S) = \mathrm{miss}_1(D,S)$ for every kept event, so the birth-window mean is exact for the kept events. (ii) For $r<1$: $\mathrm{miss}_r(D_{obs},S) \ge \mathrm{miss}_r(D,S)$, so the birth-window estimate is an upper bound; the per-event bias is at most $(1-r)^{\lfloor (W+1)/S\rfloor}$ and only censored events contribute. (iii) Under A-stat the kept events are selected on onset time only, so their durations are a sample from the onset-weighted duration law (no length bias).

*Proof.* (i) A kept censored event has $D \ge D_{obs} = L_s - b + 1 \ge W+1 > S$, so both values are 0 by Lemma 1(iii); an uncensored event has $D_{obs} = D$. (ii) $D_{obs} \le D$ and Lemma 2; the bias bound is $\mathrm{miss}_r(D_{obs},S) - \mathrm{miss}_r(D,S) \le \mathrm{miss}_r(W+1,S) \le (1-r)^{\lfloor (W+1)/S \rfloor}$. (iii) The selection indicator $\mathbb{1}\{b \le L_s - W, b>1\}$ is a function of $b$ only. ∎

*Remark (Kaplan–Meier).* For $r<1$ we also estimate $F_D$ by Kaplan–Meier using all non-left-censored events (censoring = sequence end, which is independent of $D$ under A-stat). Mass beyond the largest observed time is placed at that time; by Lemma 2 this again gives an upper bound for $\mathbb{E}[\mathrm{miss}_r(D,S)]$.

*Numerical check.* §2 Lemma 1: 12 $(D,S)$ pairs including non-integer strides (e.g. $S = 2.5, 3.7, 1.6, 7.25$), $10^5$ phases each, 48 comparisons: max $|z| = 2.10$, χ² p = 0.99; support of $N$ is exactly $\{n, n+1\}$. Lemma 1(iii): 2 700 pairs, max error 0. Lemma 2: max increment over a fine grid = 0. Corollary 1 on 400 simulated sequences with 8 % short events and administrative censoring: (i) max per-event difference 0; (ii) bias always ≥ 0.

*On data (done, P0b-A1).* UAVDT E3: 621/1 704 events (36.4 %) are right-censored. With $W = 120$ (= largest stride of the main table) the window keeps 1 462 events; $\mathrm{share\_short}(S{=}50)$ moves from 11.2 % (v1) to 8.2 % and $S = 120$ from 19.9 % to 13.6 %; Kaplan–Meier gives 8.5 % / 13.9 % (results/p0_stats.json, `window = birth_W120 | km`).

---

## 3. P1 — Kinematic certificate (B3)

### Theorem 1 (kinematic certificate)
*Assumptions.*
- **K1** Flat ground, nadir-looking camera at height $h$, field of view $\theta$ along the direction of travel; along-track footprint length $L(h) = 2h\tan(\theta/2)$. (Oblique variant: camera tilted by $\alpha$ from nadir, $|\alpha| < \pi/2 - \theta/2$: $L(h) = h[\tan(\alpha+\theta/2) - \tan(\alpha-\theta/2)]$; everything below holds with this $L$.)
- **K2** The vehicle is a point moving with constant speed $v \le v_{max}$ relative to the footprint along a straight path whose chord inside the footprint (or inside a region of interest) has length $\ell$; a full along-track traverse has $\ell = L(h)$. It is therefore in view for a continuous time interval of length $\tau = \ell/v$.
- **K3** Capture frames at times $j/F$; the detector runs on every $S$-th frame ($S$ integer), arbitrary phase: $f_{det} = F/S$.
- **A-ind** with per-look recall $r \in (0,1]$; let $n(r,\varepsilon) = \lceil \ln\varepsilon / \ln(1-r) \rceil$ for $r<1$ and $n = 1$ for $r = 1$.

*Statement.* If
$$ f_{det} \;\ge\; \frac{n(r,\varepsilon)\, v_{max}}{\ell} $$
then **for every phase** and every vehicle with $v \le v_{max}$: $N \ge n(r,\varepsilon)$ and $\mathrm{miss} \le (1-r)^{n} \le \varepsilon$. For a full traverse, $\ell = L(h)$ and the condition reads $f_{det} \ge n\,v_{max}/(2h\tan(\theta/2))$.

*Proof.* A half-open time interval of length $\tau$ contains at least $\lfloor \tau F\rfloor$ consecutive capture frames. Among any $m$ consecutive frames a period-$S$ schedule runs at least $\lfloor m/S\rfloor$ times, whatever its phase. Hence $N \ge \lfloor \lfloor \tau F \rfloor / S\rfloor = \lfloor \tau F/S \rfloor = \lfloor \tau f_{det}\rfloor$ (for integer $S$, $\lfloor \lfloor x\rfloor /S \rfloor = \lfloor x/S \rfloor$). With $\tau = \ell/v \ge \ell/v_{max}$ and the hypothesis, $\tau f_{det} \ge n$, so $N \ge n$. Under A-ind, $\mathrm{miss} = (1-r)^N \le (1-r)^n \le \varepsilon$ by the definition of $n$. ∎

*Remarks.*
1. **Camera fps drops out.** $F$ enters only through $f_{det} = F/S$: the requirement is on detector runs per second. A higher capture rate only refines the set of attainable rates $\{F/S\}$ (the smallest attainable rate above the requirement overshoots it by less than a factor $1 + f_{det}/F$). This is the design rule: *detector Hz ≥ n · v_max / chord*.
2. **Average-case version.** Under A-phase, $\mathbb{E}[\mathrm{miss}] = \mathrm{miss}_r(\tau F, S)$ exactly (Lemma 1 in continuous time).
3. **Chords, borders and the ROI.** Vehicles that clip a corner of the footprint have $\ell \to 0$ and admit no finite certificate. The certificate must therefore be stated for a region of interest whose traffic crosses it with chord $\ge \ell_{min}$ (e.g. a road segment). The data analogue is the ROI sensitivity (P0b-A2): with a 5 % inner margin the short tail $D<8$ drops from 4.1 % to 2.5 % of UAVDT E3 events and with 10 % to 2.0 %.

### Corollary 1.3 (latency version)
*Statement.* Under A-phase and A-ind, for a periodic schedule with integer stride $S$ and an event of $D$ frames,
$$P(L \le L_{max}) = 1 - \mathrm{miss}_r\bigl(\min(D, L_{max}+1),\, S\bigr).$$
In particular, for $r=1$: $P(L \le L_{max}) = 1$ whenever $S \le L_{max}+1$ and $D \ge S$. In physical units: latency $\le L_{max}/F$ seconds for every vehicle with dwell $\ge S$ frames if $f_{det} \ge F/(L_{max}+1)$.

*Proof.* $L \le L_{max}$ iff some successful look falls on one of the first $\min(D, L_{max}+1)$ frames of the event; apply Lemma 1 to that window. For $r = 1$, $S \le L_{max}+1$ and $D \ge S$ imply $S \le \min(D, L_{max}+1)$, so $\mathrm{miss}_1 = (1 - \min(D,L_{max}+1)/S)_+ = 0$. ∎

*Numerical check.* §3 Thm P1: $L = 115.5$ m ($h = 100$ m, $\theta = 60°$), $v_{max} = 25$ m/s, $\varepsilon = 1\%$, $r \in \{0.5, 0.8, 0.95\}$ ($n = 7, 3, 2$), $F \in \{10, 15, 30, 60, 120\}$ fps, 20 000 vehicles each: the minimum look count equals $n$ in all 15 settings (independent of $F$) and the Monte-Carlo miss never exceeds $\varepsilon$ (max 0.26 %). Example: $r = 0.8$ needs $f_{det} \ge 0.650$ Hz for full traverses. §3 Cor. P1-lat: 315 comparisons ($D \le 40$, $S \le 20$, $L_{max} \in\{3,5,10\}$, $r \in \{1, 0.8, 0.5\}$), $5\cdot10^4$ phases each: the $r=1$ certainty cases hold exactly; max $|z| = 3.002$ (1 exceedance, 0.85 expected), χ² p = 0.9999 → **FAIL under the literal 3σ rule** (see header).

*On data.* (done, P0b-A4, UAVDT E3, birth window, $r=1$): $P(L \le 3) \ge 0.95$ needs $S \le 4$; $L\le5$: $S \le 6$; $L \le 10$: $S\le 11$; $L \le 30$: $S \le 20$ (CI in `results/p0/latency.csv`). (P2) Replace $r$ by the detector's measured per-look recall per size bin and check the certificate against the empirical miss per speed tertile; in physical units only on UAVDT (VisDrone fps unknown).

---

## 4. P2 — Optimal altitude (B4)

Let the per-look recall depend on altitude, $r = r(h)$, and define $g(h) = -\ln(1 - r(h))$ and $\Psi(h) = h\, g(h)$. Replacing the integer $n$ in Theorem 1 by its continuous relaxation $\ln(1/\varepsilon)/g(h)$, the required detector rate for full traverses is
$$\tilde f_{det}(h) = \frac{v_{max}\ln(1/\varepsilon)}{2\tan(\theta/2)\,\Psi(h)} .$$
Minimising the detector rate is maximising $\Psi$.

### Theorem 2 (existence and first-order condition)
*Assumptions.* **H1** $r:(0,\infty)\to(0,1)$ continuous. **H2** $\Psi(h) \to 0$ as $h \to 0^+$ and as $h \to \infty$.

*Statement.* (i) $\tilde f_{det}$ attains a global minimum at some $h^* \in (0,\infty)$. (ii) If $r$ is differentiable, every interior extremum satisfies
$$\frac{d \ln g}{d \ln h}\Big|_{h^*} = -1, \qquad \text{and for small } r:\ \frac{d\ln r}{d \ln h}\Big|_{h^*} \approx -1 .$$
(iii) The integer requirement $f_{det}(h) = \lceil \ln\varepsilon/\ln(1-r(h))\rceil\, v_{max}/(2\tan(\theta/2)h)$ also attains a global minimum.

*Proof.* (i) $\Psi$ is continuous and positive. Fix $h_0$. By H2 there are $0<a<h_0<b$ with $\Psi(h) < \Psi(h_0)$ for $h \notin [a,b]$. $\Psi$ attains its maximum on the compact $[a,b]$, and this maximum is global. (ii) $\Psi' = g + h g' = 0 \iff h g'/g = -1$. Since $g = r + O(r^2)$, $d\ln g/d\ln h = d\ln r/d\ln h\,(1 + O(r))$. (iii) $x(h) = \ln\varepsilon/\ln(1-r(h))$ is continuous and $\lceil\cdot\rceil$ is lower semicontinuous, so $f_{det}$ is lower semicontinuous; moreover $f_{det} \ge \tilde f_{det} \to \infty$ at both ends by H2, so its sublevel sets are compact and the minimum is attained. ∎

### 4.3 Where the planned statement fails (counterexamples, correction)
The plan assumed "$r$ decreasing, $r\to0$ as $h\to\infty$" suffices. It does not. H2 at $h\to\infty$ requires $h\,r(h) \to 0$, i.e. **recall must decay faster than $1/h$**:
- If $r(h) \sim C h^{-\beta}$ with $\beta < 1$, then $\Psi \to \infty$: the required rate decreases for ever — "fly higher" is optimal within the model; with $\beta = 1$ an interior optimum is not guaranteed.
- If $r$ has a floor $r_0 > 0$ (e.g. a logistic curve in object pixel size, $r = r_{max}/(1+e^{-(s-s_0)/w})$ with $s = \kappa/h$, which tends to $r_{max}/(1+e^{s_0/w}) > 0$ as $s\to0$), then $\Psi \sim h\, g(r_0) \to \infty$: there may be a local optimum but no global one.

**Corrected statement (P2).** An optimal altitude exists whenever recall at high altitude vanishes faster than $1/h$ and stays below 1 at low altitude; it is characterised by the elasticity condition $d\ln(-\ln(1-r))/d\ln h = -1$. Otherwise the kinematic gain of altitude dominates and the optimum is the highest admissible altitude.

**Example (log-logistic in pixel size).** $s(h) = \kappa/h$ pixels, $r(h) = r_{max}/(1 + (s_0/s(h))^{\beta}) = r_{max}/(1 + (s_0 h/\kappa)^\beta)$. At $h\to0$, $r \to r_{max}<1$ so $\Psi\to0$; at $h \to\infty$, $r \sim r_{max}(\kappa/(s_0 h))^\beta$, so H2 holds iff $\beta > 1$. Small-$r$ approximation: with $x = (s_0 h/\kappa)^\beta$, $d\ln r/d\ln h = -\beta x/(1+x) = -1 \iff x = 1/(\beta - 1)$, i.e. $h^* \approx (\kappa/s_0)(\beta-1)^{-1/\beta}$.

*Numerical check.* §4 ($\kappa = 1500$ px·m, i.e. 30 px at 50 m; $s_0 = 12$ px; $r_{max} = 0.9$): $\beta = 3$: exact $h^* = 69.6$ m, located identically by grid maximisation and by the root of the exact first-order condition; the small-$r$ formula gives 99.2 m (biased because $r(h^*)$ is not small). $\beta = 1.5$: $h^* = 117.5$ m (approximation 198.4 m). $\beta \in \{0.8, 1.0\}$: no interior optimum on $[1, 2000]$ m. Logistic with floor ($r_0 = 0.016$): local maximum of $\Psi$ at 71.0 m but $\Psi$ increases without bound in the tail. Integer version ($\varepsilon = 1\%$, $\beta = 3$): minimum at 66.0 m with $n = 3$.

*On data (P2).* UAVDT has three altitude classes; measure $r$ per class and per object-size bin from the detector dump, and $D$ per class from annotations (P0: $D$ median low 144 vs high 358 frames). The test is the sign of the empirical elasticity between adjacent classes relative to −1. Three coarse classes (only 3 high-altitude sequences) make this a weak test; report it as such.

---

## 5. Gate channel (B5)

### 5.1 Model
- Onset windows: event $e$ has onset window $O_e = [b_e, b_e+w)$; $I_t = 1$ if frame $t$ lies in some $O_e$. $\rho = \rho_{onset}(w)$ is the long-run fraction of frames with $I_t = 1$ (measured in P0b-A5).
- **A-gate**: a frame-level cue fires $G_t \sim \mathrm{Bernoulli}(q_{in})$ if $I_t=1$ and $\mathrm{Bernoulli}(q_{out})$ otherwise, independently across frames given $I$.
- Refresh: the detector also runs on frames $t \equiv \varphi \pmod R$, $\varphi$ uniform on $\{0,\dots,R-1\}$ ($R = \infty$: no refresh), independent of everything else.
- Policy (gate ∨ refresh): run the detector on frame $t$ iff $G_t = 1$ or $t$ is a refresh frame. The cue itself costs $c$ detector-equivalents per frame (c = t_cue / t_detector, measured in P2).
- KPI window of an event: $T_e = \min(D_e, L_{max}+1)$ for the latency KPI, $T_e = D_e$ for the miss KPI. The first $w' = \min(w, T_e)$ frames are the event's own onset window.
- **A-sparse**: the remaining $T_e - w'$ frames of the KPI window are outside all onset windows. (If windows of other events overlap them and $q_{in} \ge q_{out}$, the gate only does better, so the gate miss below is then an upper bound; the full-sequence simulation in §5.6 does not impose A-sparse.)
- The periodic competitor at equal cost has activation $a_P = a_G + c$ (stride $1/a_P$, real-valued, §1).

### Proposition G1 (activation)
$$a_G = \frac1R + \Bigl(1 - \frac1R\Bigr)\bigl[\rho\, q_{in} + (1-\rho)\, q_{out}\bigr].$$
*Proof.* For any frame, $P(\text{run}_t) = P(\text{refresh}_t) + P(\neg\text{refresh}_t)P(G_t = 1) = 1/R + (1-1/R) q_t$ with $q_t = q_{in}$ or $q_{out}$, by independence of the refresh phase and the cue. Averaging over frames, the fraction with $q_t = q_{in}$ is $\rho$. ∎ (The formula in the prompt is the same expression.)

### Proposition G2 (miss of gate ∨ refresh)
Let $n_{in}(\varphi), n_{out}(\varphi)$ be the numbers of refresh frames among the first $w'$ frames and among the remaining $T_e - w'$ frames of the KPI window. Under A-gate, A-sparse, A-ind:
$$\mathrm{miss}_G(T_e) = \frac1R\sum_{\varphi=0}^{R-1} (1-r)^{n_{in}+n_{out}}\,(1 - r q_{in})^{w' - n_{in}}\,(1 - r q_{out})^{T_e - w' - n_{out}} .$$
For $r = 1$ and $T_e \le R$: $\mathrm{miss}_G = (1 - T_e/R)\,(1-q_{in})^{w'}(1-q_{out})^{T_e-w'}$. For $R = \infty$: $\mathrm{miss}_G = (1 - r q_{in})^{w'}(1-rq_{out})^{T_e - w'}$.

*Proof.* Given $\varphi$, a refresh frame yields a success with probability $r$; a non-refresh frame with cue probability $q_t$ yields a run with probability $q_t$ and then a success with probability $r$, i.e. success probability $rq_t$; frames are independent (A-gate, A-ind). The miss probability is the product of per-frame failure probabilities; average over $\varphi$. For $r = 1$, any refresh frame in the window gives a detection, and $P(\text{no refresh in } T_e \text{ frames}) = 1 - T_e/R$ for $T_e \le R$ (Lemma 1(iii) with stride $R$). ∎

### Remark G2′ (multi-look advantage under imperfect detectors) — added v1.1 (P2B)
*Setting.* Latency KPI with window $T := T_e$, gate of Prop. G2 with $R = \infty$, fixed cue parameters $(q_{in}, q_{out}, w)$ and hence a fixed activation $a = a_G + c$ that does **not** depend on the detector recall $r$ (the cue decides when the detector runs; $r$ only decides whether a run succeeds). Periodic competitor at the same activation, in the regime $Ta < 1$ (it places fewer than one look in the KPI window on average). Write $w' = \min(w,T)$ and
$$\Delta(r) = \mathrm{miss}_P(r) - \mathrm{miss}_G(r), \qquad \mathrm{miss}_P(r) = 1 - Ta\,r, \qquad \mathrm{miss}_G(r) = (1 - rq_{in})^{w'}(1 - rq_{out})^{T - w'} .$$
*Look counts.* The periodic schedule gives $N_P \in \{0, 1\}$ looks with $\mathbb E N_P = Ta$; the gate gives $\mathbb E N_G = w' q_{in} + (T - w') q_{out}$ looks, several of them consecutive. With $r = 1$ one look suffices, so extra gate looks are wasted; with $r < 1$ each extra look is another chance.

*Statement.* (i) $\Delta$ is concave on $[0,1]$ with $\Delta(0) = 0$. (ii) Hence there is $r^\dagger \in [0,1]$ such that $\Delta$ is non-increasing in $r$ on $[r^\dagger, 1]$ — **the gate's advantage grows as the detector gets worse, down to $r^\dagger$** — and non-decreasing on $[0, r^\dagger]$ (the advantage vanishes as $r \to 0$). (iii) $r^\dagger < 1$ iff $\Delta'(1) < 0$, i.e. iff
$$ T a \;>\; -\mathrm{miss}_G'(1) = \mathrm{miss}_G(1)\Bigl[\tfrac{w' q_{in}}{1 - q_{in}} + \tfrac{(T - w')q_{out}}{1 - q_{out}}\Bigr] \quad (q_{in} < 1), $$
which always holds when $q_{in} = 1$ and $w' \ge 2$ (then $\mathrm{miss}_G'(1) = 0$). (iv) For $q_{out} = 0$ and $w' = T$: $r^\dagger = \min\{1,\,(1 - (a/q_{in})^{1/(T-1)})/q_{in}\}$.

*Proof.* $\mathrm{miss}_P$ is affine in $r$ by Lemma 1 ($n = 0$, $f = Ta$). Each factor $(1 - rq)^k$ is non-negative, non-increasing and convex in $r$; for two such functions $(fg)'' = f''g + 2f'g' + fg'' \ge 0$ because $f', g' \le 0$, so $\mathrm{miss}_G$ is convex and $\Delta$ is concave. $\Delta(0) = 1 - 1 = 0$. A concave function on $[0,1]$ has a (possibly boundary) maximiser $r^\dagger$, is non-decreasing to its left and non-increasing to its right, which is (ii); (iii) is the condition that the maximiser is not at $r = 1$. For (iv), $\Delta'(r) = -Ta + T q_{in}(1 - rq_{in})^{T-1} = 0$ gives the stated root, truncated to $[0,1]$. ∎

*Scope (where the naive reading fails).* The claim "a worse detector always favours the gate" is **false** in general: (a) below $r^\dagger$ the advantage decreases with $r$; (b) if $q_{in}$ is small, $r^\dagger = 1$ and $\Delta$ decreases as soon as $r$ drops (example: $T = 4$, $w' = 4$, $q_{in} = 0.5$, $q_{out} = 0$, $a = 0.05$: $\Delta'(1) = +0.05 > 0$); (c) in the regime $Ta \ge 1$, $\mathrm{miss}_P(r) = (1-r)^n(1 - fr)$ is convex too and the difference of two convex functions has no general shape. Averaging over events with different $T_e$ preserves (i)–(ii) as long as every event is in the regime $T_e a < 1$ (a sum of concave functions vanishing at 0).

*Numerical check.* §5 Remark G2′ (theory_checks.json): 5 settings × $r \in \{1, 0.8, 0.5, 0.3\}$, $4\cdot10^4$ single-event simulations each (lattice periodic + Bernoulli gate): max $|z| = 1.75$ (20 comparisons, χ² p = 0.57); concavity and $\Delta(0) = 0$ hold on a $10^5$-point grid; the closed form (iv) matches the numerical maximiser to $6\cdot10^{-7}$; condition (iii) predicts $r^\dagger < 1$ correctly in 5/5 settings (4 interior peaks, 1 boundary case = counterexample (b)). On the A7 replay (UAVDT TRAIN, simulated detector, 80 cue configurations × 3 latency KPIs × 3 pairs of $r$): the predicted sign of $\Delta(r_{lo}) - \Delta(r_{hi})$ matches the replay in 97.5 % of 720 pairs and in 100 % of the 533 pairs whose conservative interval excludes 0; "Δ increases as $r$ decreases" holds in 72.9 % of pairs (predicted 73.5 %) — conditional, as stated (results/p2/sim/multilook.json).

*On data (P2, preregistered as H8).* With real detectors, lower recall comes from yolo26n vs yolo26s and from 640 vs 1280 input. Prediction: for cue configurations with $r^\dagger$ (computed from the TRAIN channel) below both detectors' recalls, Δ(latency) is larger for the lower-recall detector.

### Theorem G3 (gain–cost identity, $r = 1$)
Let $T := T_e \le R$, $w' \ge 1$ and $\Delta := \mathrm{miss}_P - \mathrm{miss}_G$ with $\mathrm{miss}_P$ the periodic miss at activation $a_G + c$. If $T(a_G + c) \ge 1$ then $\Delta = -\mathrm{miss}_G \le 0$. Otherwise
$$\Delta = \underbrace{\Bigl(1 - \tfrac{T}{R}\Bigr)\Bigl[1 - (1-q_{in})^{w'}(1-q_{out})^{T - w'}\Bigr]}_{\text{gain } \mathcal G} \;-\; \underbrace{T\Bigl(1 - \tfrac1R\Bigr)\bigl[\rho q_{in} + (1-\rho) q_{out}\bigr]}_{\text{cost } \mathcal C} \;-\; T c .$$
*Interpretation.* $\mathcal G$ is the probability that the gate catches an event missed by refresh; $\mathcal C + Tc$ is the probability that a periodic schedule would have caught it with the budget the gate spends on cue firings and on the cue itself.

*Proof.* By Lemma 1(iii) with stride $1/(a_G+c)$, $\mathrm{miss}_P = (1 - T(a_G + c))_+$. If $T(a_G+c)\ge1$ this is 0. Otherwise substitute Proposition G1 and G2 ($r=1$): $\mathrm{miss}_P - \mathrm{miss}_G = 1 - T/R - T(1 - 1/R)[\rho q_{in} + (1-\rho)q_{out}] - Tc - (1 - T/R)(1-q_{in})^{w'}(1-q_{out})^{T-w'}$, which rearranges to the display. ∎

(If $T > R$, refresh alone guarantees a look in the window: $\mathrm{miss}_G = 0 = \mathrm{miss}_P$ since $a_G \ge 1/R > 1/T$; no strict win is possible.)

### Corollary G4 (threshold structure, $r = 1$)
Fix $(q_{out}, \rho, R, w, T, c)$ with $T \le R$ and let $\kappa(R,T) = \dfrac{1 - T/R}{T(1 - 1/R)} = \dfrac{R-T}{T(R-1)}$ ($\kappa = 1/T$ for $R=\infty$).
1. **Interval.** The set of $q_{in}\in[0,1]$ for which the gate strictly wins is an interval $(q_{in}^*, q_{in}^{up})$ (possibly empty, possibly with $q_{in}^{up} = 1$ included). The threshold asked for in the plan is $q_{in}^*(q_{out},\rho,R)$; "$q_{in}^* > 1$" denotes the empty set.
2. **Necessary condition (N-G).** A win requires $(1-\rho)\,q_{out} + \dfrac{c}{1 - 1/R} < \kappa(R,T)$.
3. **Sufficient condition (S-G).** If $\rho + (1-\rho) q_{out} + \dfrac{c}{1-1/R} < \kappa(R,T)$, the gate with $q_{in}=1$ wins.
4. **First order (R = ∞).** For small $q_{in}, q_{out}$: $\Delta \approx (q_{in} - q_{out})(w' - T\rho) - Tc$, hence $q_{in}^* \approx q_{out} + Tc/(w' - T\rho)$ when $\rho < w'/T$, and no small-$q$ win when $\rho \ge w'/T$.

*Proof.* (1) $a_G$ is increasing in $q_{in}$, so $\{q_{in}: T(a_G+c) \le 1\} = [0, q_c]$ is an interval, and outside it $\Delta \le 0$. On it, $\Delta = \mathcal G - \mathcal C - Tc$ with $\mathcal G$ concave in $q_{in}$ (since $(1-q_{in})^{w'}$ is convex) and $\mathcal C$ affine, so $\Delta$ is concave and $\{\Delta > 0\}$ is an interval. (2) $\mathcal G \le 1 - T/R$ and $\mathcal C \ge T(1-1/R)(1-\rho)q_{out}$; $\Delta > 0$ forces $T(1-1/R)(1-\rho)q_{out} + Tc < 1 - T/R$. (3) At $q_{in} = 1$, $\mathrm{miss}_G = 0$ (since $w'\ge1$) and $\Delta = 1 - T(a_G(1) + c) > 0$ under the stated inequality. (4) With $R=\infty$, $1 - (1-q_{in})^{w'}(1-q_{out})^{T-w'} = w' q_{in} + (T - w') q_{out} + O(q^2)$ and $\mathcal C = T[\rho q_{in} + (1-\rho)q_{out}]$; subtract. ∎

### 5.4 Where the planned statement fails, and the corrected reading
The plan expected a single threshold "$q_{in}^*$ such that the gate wins iff $q_{in} > q_{in}^*$". Two corrections:
- The win set is an interval that can be **bounded above**: when $\rho$ is not small, firing on every onset-window frame costs more activation than it gains (e.g. $L_{max} = 5$, $w=5$, $R = 30$, $\rho = 0.160$, $q_{out} = 0.05$, $r=1$: win set ≈ (0.091, 0.589)).
- At **matched activation in the low-budget regime** ($a < 1/T$) almost any informative cue wins (first order: $q_{in} > q_{out}$), because a periodic schedule with $a < 1/T$ misses a fraction $1 - Ta$ of short windows. This is true but not the operating regime of an ITS deployment, where the budget is set by a target miss. The ITS-relevant comparison is **iso-KPI** (Corollary G5).

### Corollary G5 (iso-KPI, $r = 1$, ideal onset gate)
*Assumptions:* events with $D \ge T$; onset gate with $w = 1$ and $q_{in} = 1$ (fires on every onset frame), false-fire rate $q_{out}$, no refresh; $\rho_1 = \rho_{onset}(1)$ (≈ onset rate $\lambda$ per frame when onsets rarely coincide).

*Statement.* To reach latency-miss $\le \varepsilon$ the periodic schedule needs $a_P = (1-\varepsilon)/T$; the ideal onset gate reaches latency-miss 0 at total cost $a_G + c = \rho_1 + (1-\rho_1)q_{out} + c$. Hence the gate is cheaper at equal KPI iff
$$ \rho_1 + (1-\rho_1)\, q_{out} + c \;<\; \frac{1-\varepsilon}{T}. $$
*Proof.* Periodic: Lemma 1(iii) with $D \ge T$: $\mathrm{miss} = 1 - Ta$, set equal to $\varepsilon$. Gate: the onset frame is always looked at and $r = 1$; cost by Proposition G1 with $R = \infty$, $w=1$. ∎

For $r<1$ the onset gate must fire on at least $n(r,\varepsilon)$ window frames, which multiplies the $\rho$ term accordingly; the comparison is then numerical.

### 5.6 Numerical checks
§5 Prop. G1: 4 settings × 30 sequences of 20 000 frames with Poisson onsets: max $|z| = 1.72$. Prop. G2: 15 settings ($r \in \{1, 0.8, 0.5\}$, finite and infinite $R$), $6\cdot10^4$ events each: max $|z| = 2.71$. Thm G3: 2 970 random configurations, max error $6.7\cdot10^{-16}$. Cor. G4: 3 000 random configurations — the win set was an interval in all, and no winning configuration violated (N-G). Full-sequence simulation without A-sparse (overlapping windows, real refresh lattice, fractional-stride periodic): predicted Δ = +0.807 vs simulated +0.818 ± 0.007 (win cell) and −0.00696 vs −0.00703 ± 0.0005 (lose cell). Cor. G5: 3 settings, activation and periodic miss match (max $|z| = 1.44$), gate miss 0.

*On data (P2).* Estimate $q_{in}, q_{out}, c$ per cue on the TRAIN split, $\rho_{onset}(w)$ from annotations (P0b-A5: UAVDT pooled 9.9 % / 16.0 % / 29.5 % for $w$ = 3 / 5 / 10 frames), then predict the sign of Δ per cell (PREREG draft) and compare with the replayed policies on TEST.

---

## 6. P3 — Density and ego-motion impossibility (B6)

### Lemma P3.1 (effective false-fire rate)
*Assumptions:* on a frame outside onset windows the cue fires if the background/ego-motion residual fires (probability $q_b$) or if any of $M$ other moving objects triggers it (probability $q_o$ each), all independent.
*Statement.* $q_{out,eff}(M) = 1 - (1-q_b)(1-q_o)^M$. *Proof.* The cue stays silent iff none of the $M+1$ independent sources fires. ∎

### Theorem P3 (no frame-level gate wins beyond a density or ego-motion threshold)
Let $r = 1$ and $T \le R$.

(i) **Density.** Define $q^{max}_{out} = \bigl(\kappa(R,T) - c/(1-1/R)\bigr)/(1-\rho)$. If $q_b < q^{max}_{out} < 1$, then for every
$$ M \;\ge\; M_{nec} := \frac{\ln\bigl((1-q_b)/(1-q^{max}_{out})\bigr)}{-\ln(1-q_o)} $$
no choice of $q_{in} \in [0,1]$ makes the gate beat the periodic schedule of equal cost. Hence $M^* := \min\{M : \text{no gate wins for any } M' \ge M\}$ exists and $M^* \le \lceil M_{nec}\rceil$. If $q_b \ge q^{max}_{out}$, no gate wins at any $M$ ($M^* = 0$).

(ii) **Ego-motion.** Without compensation, camera motion makes the background residual fire on almost every frame, $q_b \to 1$, so $q_{out,eff}\to1$ and (N-G) fails for every $M$: no frame-level gate wins. With compensation at cost $c$ per frame, (N-G) requires $(1-\rho)q_{out,eff} + c/(1-1/R) < \kappa$; and for any total budget $a < c + a_{min}$, $a_{min} = 1/R + (1-1/R)(1-\rho) q_{out,eff}$, no gate configuration is even feasible while a periodic schedule is.

(iii) **Iso-KPI version.** For the ideal onset gate of Corollary G5 the gate is cheaper at equal latency-miss $\varepsilon$ iff $q_{out,eff}(M) < Q := \bigl((1-\varepsilon)/T - \rho_1 - c\bigr)/(1-\rho_1)$, i.e. iff $M < M_{iso} := \ln\bigl((1-q_b)/(1-Q)\bigr)/(-\ln(1-q_o))$ (and never if $Q \le q_b$).

*Proof.* (i) $q_{out,eff}(M)$ is increasing in $M$ and $q_{out,eff}(M) \ge q^{max}_{out} \iff M \ge M_{nec}$ (solve Lemma P3.1). For such $M$, (N-G) of Corollary G4 fails, so no $q_{in}$ wins. (ii) $q_b \to 1$ gives $q_{out,eff} \to 1 > q^{max}_{out}$ whenever $\kappa < 1-\rho$, which holds for all $T \ge 2$ unless $\rho > 1 - 1/T$. The feasibility bound is Proposition G1 at $q_{in} = 0$ plus the cue cost. (iii) Corollary G5 with Lemma P3.1, solved for $M$. ∎

Note that $\Delta$ need not be monotone in $q_{out}$ (extra out-of-window firings inside a long KPI window can help), so the exact $M^*$ is computed numerically; the closed form $M_{nec}$ is an upper bound.

### Numerical illustration (UAVDT; $q_b, q_o, c$ are illustrative until measured on TRAIN in P2)
UAVDT: median number of visible vehicles $M = 14$; onset rate $\lambda = 0.0422$ per frame; $\rho_{onset}$ = 0.099 / 0.160 / 0.295 for $w$ = 3 / 5 / 10.

Matched activation ($r = 1$, $R=\infty$, $q_b = 0$, $c = 0$), exact $M^*$ (bound $M_{nec}$):

| $L_{max}$ ($T$, $w$) | $q_o$ = 0.005 | 0.01 | 0.02 | 0.05 |
|---|---|---|---|---|
| 3 (4, 3) | 44 (64.8) | 22 (32.3) | 11 (16.1) | 5 (6.3) |
| 5 (6, 5) | 24 (44.1) | 12 (22.0) | 6 (10.9) | 3 (4.3) |
| 10 (11, 10) | 10 (27.5) | 5 (13.7) | 3 (6.8) | 1 (2.7) |

Iso-KPI ($\varepsilon = 1\%$, $q_b = 0$, $c = 0$), $M_{iso}$: $L_{max} = 3$: 48.1 / 24.0 / 11.9 for $q_o$ = 0.005 / 0.01 / 0.02; $L_{max} = 5$: 27.4 / 13.7 / 6.8; $L_{max}=10$: 10.2 / 5.1 / 2.5; $L_{max} = 30$: **0** — at $T = 31$ the periodic schedule needs only $a = 0.032 < \lambda$, so even a perfect onset gate cannot be cheaper.

Reading: at UAVDT's median density the boundary falls exactly in the plausible range of the per-object false-fire probability ($q_o \approx 0.01$–0.02); only tight latency targets ($L_{max} \le 5$ frames ≈ 0.17 s at 30 fps) leave room for a gate, and only in sparse traffic or with a cue that is nearly blind to other vehicles (region-level gating). This is the quantitative content of P3.

*Numerical check.* §6 Lemma P3.1: 4 settings, $2\cdot10^5$ frames each, max $|z| = 1.84$. Thm P3(i): 54 settings, exact $M^* \le \lceil M_{nec}\rceil$ in all. Thm P3(ii): $q_b = 0.999$ ⇒ no win in all tested cells; $c \in \{0.2, 0.3\}$ ⇒ no win.

*On data (P2).* Measure $q_b$ per sequence (hovering vs moving, with and without ORB compensation), $q_o$ by regressing cue firing outside onset windows on $M(t)$ (Lemma P3.1 predicts $\ln(1 - q_{out}) = \ln(1-q_b) + M\ln(1-q_o)$, linear in $M$), and $c$ from the laptop benchmark; then check that cells with $M > M^*$ show $\Delta \le 0$.

---

## 7. Summary of statements and domains of validity

| # | Statement | Assumptions | Domain / caveat | Check |
|---|---|---|---|---|
| L1 | $N = \lfloor D/S\rfloor + \mathrm{Bern}(f)$, $\mathrm{miss}_r = (1-r)^n(1-fr)$ | A-phase, A-ind | exact; real or integer $S$ | PASS |
| L2 | miss non-increasing in $D$ | — | exact | PASS |
| C1 | birth window exact for $r=1$, upper bound for $r<1$ | A-stat | needs $S \le W$ | PASS |
| P1 | $f_{det} \ge n v_{max}/\ell$ ⇒ miss ≤ ε for every phase | K1–K3, A-ind | chord $\ell$ must be bounded below (ROI) | PASS |
| P1-lat | $P(L\le L_{max}) = 1 - \mathrm{miss}_r(\min(D,L_{max}+1),S)$ | A-phase, A-ind | exact | PASS (family-wise; 1/315 at 3.002σ, χ² p = 0.9999) |
| P2 | optimal altitude exists, elasticity −1 | H1, H2 | **only if $h\,r(h)\to0$**; false for recall floors or decay ≤ $1/h$ | PASS (incl. counterexamples) |
| G1–G3 | activation, gate miss, gain–cost identity | A-gate, A-sparse, A-ind | G3 for $r=1$, $T\le R$ | PASS |
| G2′ | multi-look: Δ(r) concave, Δ(0)=0, gate advantage grows as $r$ falls on $[r^\dagger,1]$ | as G2, $R=\infty$, $Ta<1$ | **conditional**: $r^\dagger<1$ iff $Ta>-\mathrm{miss}_G'(1)$; false below $r^\dagger$ | PASS |
| G4 | win set is an interval; (N-G), (S-G), first-order threshold | as G3 | interval may be bounded above | PASS |
| G5 | iso-KPI condition $\rho_1 + (1-\rho_1)q_{out} + c < (1-\varepsilon)/T$ | $r=1$, $D \ge T$ | for $r<1$ numerical | PASS |
| P3 | $M \ge M_{nec}$ or uncompensated ego-motion ⇒ no frame-level gate wins | Lemma P3.1 independence | $r=1$ closed form; $r<1$ numerical | PASS |

## References
[Deep3] Deep3 manuscript, under review at IEEE Internet of Things Journal (R1 submitted 09/2026); full reference to be filled from the published version. Cited only as a related derivation of a gain–cost threshold for frame-level gating — no result here depends on it.
