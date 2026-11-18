function s = mem_ridge(x, y, q, p)
% Kernel ridge classification. All scaling comes from labeled training rows.
mu = mean(x, 1);
sd = std(x, 0, 1);
ok = isfinite(mu) & sd > 1e-7;
if ~any(ok)
    s = repmat(mean(y, 1), size(q, 1), 1);
    return
end
sd = max(sd(ok), max(0.30 * median(sd(ok)), 1e-5));
x = double((x(:, ok) - mu(ok)) ./ sd);
q = double((q(:, ok) - mu(ok)) ./ sd);
x = max(-8, min(8, x));
q = max(-8, min(8, q));
y = double(y);
bias = mean(y, 1);
reg = p.lam * eye(size(x, 1));
if p.rel == 0
    [k, kq] = gram(x, q, p.rbf);
    s = kq * ((k + reg) \ (y - bias)) + bias;
    return
end

% A small preference for features which distinguish the two training classes.
s = zeros(size(q, 1), size(y, 2));
for j = 1:size(y, 2)
    high = y(:, j) == 1;
    if ~any(high) || all(high)
        s(:, j) = bias(j);
        continue
    end
    diff = mean(x(high, :), 1) - mean(x(~high, :), 1);
    w = sqrt(1 + p.rel * min(3, abs(diff) / .40).^2);
    [k, kq] = gram(x .* w, q .* w, p.rbf);
    s(:, j) = kq * ((k + reg) \ (y(:, j) - bias(j))) + bias(j);
end
end

function [k, kq] = gram(x, q, rbf)
z = max(mean(sum(x.^2, 2)), 1e-6);
k = x * x.' / z;
kq = q * x.' / z;
if rbf > 0
    dx = sum(x.^2, 2) / z;
    dq = sum(q.^2, 2) / z;
    d = max(0, dx + dx.' - 2*k);
    dQ = max(0, dq + dx.' - 2*kq);
    k = (1-rbf)*k + rbf*exp(-d);
    kq = (1-rbf)*kq + rbf*exp(-dQ);
end
end
