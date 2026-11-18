function r = mem_labfit(c, cfg)
% Evaluate fixed families and choose new models using only inner folds.
[~, ~, b] = mem_predict(c, cfg);
[z, names] = trialModels(c, cfg);
n = numel(c.cal.id);
q = {[b.score(:, 1), b.cueScore, z{1}], [b.score(:, 2), z{2}]};
r.names = {["v2", "deduplicated_v2", "target_only_v2", names{1}], ["v2", names{2}]};

% The fixed policy was proposed in the separate Sol candidate experiments.
fixed = [1, find(r.names{2} == behaviorName(c.qry), 1)];
early = fixed;
source = fixed;
if c.qry.ch == 2
    early(1) = find(r.names{1} == "early_unit_ridge", 1);
    source(1) = find(r.names{1} == "early_with_matched_source", 1);
end

fold = mem_folds(c.cal, cfg);
cv = cell(1, 2);
cv{1} = cat(3, reshape(b.cv(:, 1, :), n, cfg.reps), b.cueCV, ...
    zeros(n, cfg.reps, size(z{1}, 2)));
cv{2} = cat(3, reshape(b.cv(:, 2, :), n, cfg.reps), zeros(n, cfg.reps, size(z{2}, 2)));
for i = 1:cfg.reps
    for k = 1:cfg.folds
        va = fold(:, i) == k;
        h = c;
        h.cal = mem_take(c.cal, ~va);
        h.qry = mem_take(c.cal, va);
        h.qry.y = [];
        [s, tag] = trialModels(h, cfg);
        assert(isequal(tag, names), 'Candidate order changed between folds.');
        cv{1}(va, i, 4:end) = reshape(s{1}, nnz(va), 1, []);
        cv{2}(va, i, 2:end) = reshape(s{2}, nnz(va), 1, []);
    end
end

inner = zeros(1, 2);
for j = 1:2
    y = c.cal.y(:, j);
    ok = (cv{j} >= .50) == y;
    acc = reshape(mean(ok, [1 2]), 1, []);
    prob = max(0, min(1, cv{j}));
    loss = reshape(mean((prob-y).^2, [1 2]), 1, []);
    use = acc >= acc(1) + cfg.minInnerGain / n;
    use(1) = true;
    adj = acc - .015*loss;
    adj(~use) = -inf;
    [~, inner(j)] = max(adj);
end

r.policy = ["v2", "inner_behavior", "inner_both", "fixed_behavior", ...
    "early_cue_and_behavior", "matched_source_and_behavior"];
ix = [1 1; 1 inner(2); inner; fixed; early; source];
r.pred = false(numel(c.qry.id), 2, size(ix, 1));
r.selected = strings(size(ix));
for j = 1:2
    r.pred(:, j, :) = reshape(q{j}(:, ix(:, j)) >= .50, numel(c.qry.id), 1, []);
    r.selected(:, j) = reshape(r.names{j}(ix(:, j)), [], 1);
end
r.cand = {q{1} >= .50, q{2} >= .50};
end

function [s, names] = trialModels(c, cfg)
[s{1}, names{1}] = mem_time_trials(c, cfg);
if c.qry.ch == 2
    [a, tag] = mem_cue_trials(c, cfg);
    i = tag == "early_unit_ridge";
    j = tag == "early_matched_src0.5";
    assert(nnz(i) == 1 && nnz(j) == 1);
    a = max(0, min(1, a));
    mix = .85*a(:, i) + .15*a(:, j);
    s{1} = [s{1}, a, mix];
    names{1} = [names{1}, tag, "early_with_matched_source"];
end
[s{2}, names{2}] = mem_beh_trials(c, cfg);
end

function name = behaviorName(d)
if d.ch == 1
    name = "prior";
elseif d.ch == 2
    name = "mean_sqrt";
elseif startsWith(d.name, 'sub-m090_')
    name = "cue_conditioned";
else
    name = "mean_raw";
end
end
