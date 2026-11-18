function [score, names] = mem_beh_trials(c, cfg)
% Compare behavior models fitted only on labeled trials from this session.
opt = settings(cfg);
a = c.cal;
q = c.qry;
y = a.y(:, 2);
nq = numel(q.id);
names = ["prior", "mean_raw", "mean_sqrt", "late_rate", "trajectory_rate", ...
    "population_shape", "cue_consistency", "cue_conditioned", "cue_mean", "cue_trajectory"];
score = zeros(nq, numel(names));
score(:, 1) = mean(y);

p = struct('lam', opt.lam, 'rbf', 0, 'rel', 0);
score(:, 2) = mem_ridge(a.f(:, :, 1), y, q.f(:, :, 1), p);
p.lam = opt.rate;
score(:, 3) = mem_ridge(sqrt(a.f(:, :, 1)), y, sqrt(q.f(:, :, 1)), p);
score(:, 4) = mem_ridge(sqrt(a.f(:, :, 18)), y, sqrt(q.f(:, :, 18)), p);
score(:, 5) = mem_ridge(win(a, opt.traj), y, win(q, opt.traj), p);
p.lam = opt.small;
score(:, 6) = mem_ridge(pop(a, opt.traj), y, pop(q, opt.traj), p);

% Cue scores from smaller training sets are used on both sides of this fit.
% Each labeled trial gets a score from a decoder that never saw that trial.
[u, v] = cueScores(c, opt);
z = agreement(u);
w = agreement(v);
s = mem_ridge(z, y, w, p);
score(:, 7) = mean(reshape(s, nq, []), 2);

% Fit each odor's outcome model, then average by the predicted odor score.
odor = a.y(:, 1);
if min(nnz(odor), nnz(~odor)) >= opt.minClass
    s0 = mem_ridge(z(~odor, :), y(~odor), w, p);
    s1 = mem_ridge(z(odor, :), y(odor), w, p);
    pr = mean(v, 2);
    cond = mean(reshape((1-pr).*s0 + pr.*s1, nq, []), 2);
    score(:, 8) = (1-opt.mix)*score(:, 7) + opt.mix*cond;
else
    score(:, 8) = score(:, 7);
end
score(:, 9) = (1-opt.mix)*score(:, 2) + opt.mix*score(:, 7);
score(:, 10) = (1-opt.mix)*score(:, 5) + opt.mix*score(:, 7);
score = max(0, min(1, score));
end

function opt = settings(cfg)
opt.lam = .10;  % Existing raw mean-rate comparator.
opt.rate = .12;
opt.small = .35;
opt.cue = .20;
opt.mix = .50;
opt.minClass = 5;
opt.nFold = 4;
opt.seed = 42;
opt.traj = [6 13 18]; % First 1 s, 1.35–2 s, last 1 s.
opt.odorWin = {[2 8 9 10], 13, [16 17]};
if isfield(cfg, 'seed')
    opt.seed = cfg.seed;
end
if isfield(cfg, 'beh')
    fields = fieldnames(cfg.beh);
    for j = 1:numel(fields)
        key = fields{j};
        assert(isfield(opt, key), 'Unknown behavior option %s.', key);
        opt.(key) = cfg.beh.(key);
    end
end
end

function x = win(d, ix)
x = reshape(sqrt(d.f(:, :, ix)), numel(d.id), []);
end

function x = pop(d, ix)
% Population dynamics and regional mean rates retain a small feature count.
x = [double(d.pop), double(d.dyn)];
reg = d.reg(:, 2);
labels = unique(reg(reg ~= ""));
nu = numel(reg);
for j = 1:numel(labels)
    use = reg == labels(j);
    if nnz(use) < max(12, .02*nu)
        continue
    end
    rate = reshape(mean(d.f(:, use, ix), 2), numel(d.id), []);
    x = [x, log1p(double(rate))]; %#ok<AGROW>
end
end

function [u, v] = cueScores(c, opt)
a = c.cal;
q = c.qry;
n = numel(a.id);
nq = numel(q.id);
odor = a.y(:, 1);
k = min(opt.nFold, n);
assert(k >= 2, 'Behavior cue evidence needs at least two labeled trials.');
% Stratify on cue and delay. Behavior labels never enter the cue decoder.
rs = RandStream('mt19937ar', 'Seed', opt.seed);
[~, ~, group] = unique([a.delay(:), double(odor)], 'rows');
fold = zeros(n, 1);
off = 0;
for j = 1:max(group)
    ids = find(group == j);
    ids = ids(randperm(rs, numel(ids)));
    fold(ids) = mod(off + (0:numel(ids)-1), k) + 1;
    off = off + numel(ids);
end

nw = numel(opt.odorWin);
u = zeros(n, nw);
v = zeros(nq*k, nw);
p = struct('lam', opt.cue, 'rbf', 0, 'rel', 0);
for j = 1:nw
    x = win(a, opt.odorWin{j});
    xq = win(q, opt.odorWin{j});
    for f = 1:k
        held = fold == f;
        rows = (f-1)*nq + (1:nq);
        s = mem_ridge(x(~held, :), odor(~held), [x(held, :); xq], p);
        u(held, j) = s(1:nnz(held));
        v(rows, j) = s(nnz(held)+1:end);
    end
end
u = max(0, min(1, u));
v = max(0, min(1, v));
end

function x = agreement(s)
% Sign agreement and magnitude describe the change in cue evidence over time.
u = 2*s - 1;
x = [u, abs(u), u(:, 1).*u(:, 2), u(:, 1).*u(:, 3), u(:, 2).*u(:, 3)];
end
