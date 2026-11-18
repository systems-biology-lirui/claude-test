function [score, names] = mem_time_trials(c, cfg)
% Decode odor from the delay thirds. Scores use .5 as the decision boundary.
if ~isfield(cfg, 'timeLda')
    cfg.timeLda = .90;
end
if ~isfield(cfg, 'timeRbf')
    cfg.timeRbf = 1;
end

a = c.cal;
q = c.qry;
y = a.y(:, 1) == 1;
names = ["third_first_lda", "thirds_lda", "thirds_change_lda", ...
    "onset_thirds_lda", "thirds_by_delay_lda", "thirds_rbf_center", ...
    "thirds_change_rbf_center"];
score = .5 * ones(numel(q.id), numel(names));
if ~any(y) || all(y)
    score(:, :) = double(any(y));
    return
end

x = sqrt(double(a.f(:, :, 24:26)));
v = sqrt(double(q.f(:, :, 24:26)));
[t, u] = scale(flat(x), flat(v));
nu = size(x, 2);
score(:, 1) = lda(t(:, 1:nu), y, u(:, 1:nu), cfg.timeLda);
score(:, 2) = lda(t, y, u, cfg.timeLda);
score(:, 6) = rbfCenter(t, y, u, cfg.timeRbf);

% Two adjacent differences isolate trial-specific changes over the delay.
[tx, ux] = scale(flat(diff(x, 1, 3)), flat(diff(v, 1, 3)));
score(:, 3) = lda(tx, y, ux, cfg.timeLda);
score(:, 7) = rbfCenter(tx, y, ux, cfg.timeRbf);

% The first 100 ms and the three delay thirds each contribute one block.
[tx, ux] = scale(sqrt(double(a.f(:, :, 3))), sqrt(double(q.f(:, :, 3))));
score(:, 4) = lda([tx, t], y, [ux, u], cfg.timeLda);

% Subtract each duration's baseline, estimated from calibration rows.
for d = unique(a.delay).'
    at = a.delay == d;
    mu = mean(t(at, :), 1);
    t(at, :) = t(at, :) - mu;
    u(q.delay == d, :) = u(q.delay == d, :) - mu;
end
score(:, 5) = lda(t, y, u, cfg.timeLda);
end

function x = flat(x)
x = reshape(x, size(x, 1), size(x, 2)*size(x, 3));
end

function [x, q] = scale(x, q)
mu = mean(x, 1);
sd = std(x, 0, 1);
use = sd > 1e-7;
x(:, ~use) = 0;
q(:, ~use) = 0;
if any(use)
    sd = max(sd(use), .30 * median(sd(use)));
    x(:, use) = max(-8, min(8, (x(:, use) - mu(use)) ./ sd));
    q(:, use) = max(-8, min(8, (q(:, use) - mu(use)) ./ sd));
end
end

function s = lda(x, y, q, shrink)
% Use an identity-shrunk within-class covariance. Solve in trial space.
hi = mean(x(y, :), 1);
lo = mean(x(~y, :), 1);
dir = (hi - lo).';
e = x - (hi .* y + lo .* ~y);
wt = (1-shrink) / (shrink * max(size(x, 1)-2, 1));
b = dir - wt * e.' * ((eye(size(x, 1)) + wt * e * e.') \ (e * dir));
% Normalize so the two fitted class centers score 0 and 1.
s = .5 + (q - .5*(hi+lo)) * b / max(dir.'*b, 1e-8);
end

function s = rbfCenter(x, y, q, width)
% Compare query distances to the two class centers in the RBF kernel space.
z = max(width * mean(sum(x.^2, 2)), 1e-8);
normX = sum(x.^2, 2).';
k = exp(-max(0, normX.' + normX - 2*x*x.') / z);
kq = exp(-max(0, sum(q.^2, 2) + normX - 2*q*x.') / z);
a = mean(k(y, y), 'all');
b = mean(k(~y, ~y), 'all');
gap = a + b - 2*mean(k(y, ~y), 'all');
s = .5 + (mean(kq(:, y), 2) - mean(kq(:, ~y), 2) - .5*(a-b)) / max(gap, 1e-8);
end
