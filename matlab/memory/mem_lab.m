function out = mem_lab(c, cfg)
% Same outer trial folds for every policy, plus an ordered-block check.
if ~isfolder(cfg.out)
    mkdir(cfg.out);
end
t = tic;
oldTime = 0;
if strcmp(cfg.mode, 'v3_finish')
    prior = load(fullfile(cfg.out, 'paired_trials.mat'), 'audit', 'group', 'out');
    audit = prior.audit;
    group = prior.group;
    policy = prior.out.policy;
    oldTime = prior.out.elapsed;
    for i = 1:numel(c)
        assert(isequal(audit(i).id, c(i).cal.id) && isequal(audit(i).y, c(i).cal.y), ...
            'The cached comparison uses different labeled trials.');
    end
else
    [audit, group, policy] = bench(c, cfg);
end

np = numel(policy);
acc = zeros(3, np);
detail = zeros(4, 2, np, 3);
for i = 1:numel(c)
    a = audit(i);
    for j = 1:3
        v = mean(a.pred(:, :, :, group == j) == a.y, [1 4]);
        detail(i, :, :, j) = v;
        acc(j, :) = acc(j, :) + a.weight * reshape(mean(v, 2), 1, []);
    end
end

% Cross-day submission policies use source training with target calibration.
submit = ~ismember(policy, ["inner_both", "early_cue_and_behavior"]);
delta = acc - acc(:, 1);
ok = all(delta(1:2, :) >= cfg.minPackageGain, 1) & delta(3, :) >= -1e-9;
ok(1) = true;
fit = mean(acc(1:2, :), 1);
fit(~ok | ~submit) = -inf;
[~, pick] = max(fit);

out.zip = '';
out.acc = acc;
out.detail = detail;
out.policy = policy;
out.pick = pick;
out.score = acc(1, pick);
out.base = acc(1, 1);
out.report = fullfile(cfg.out, 'validation.txt');
out.chosen = strings(4, 2);
if pick > 1
    for i = 1:numel(c)
        r = mem_labfit(c(i), labOptions(cfg, c(i).qry.ch));
        p(i) = struct('name', c(i).qry.name, 'id', c(i).qry.id, ...
            'y', r.pred(:, :, pick)); %#ok<AGROW>
        out.chosen(i, :) = r.selected(pick, :);
    end
    out.zip = mem_pack(p, cfg);
end
out.elapsed = oldTime + toc(t);
writeReport(out, audit, cfg, group, ok, submit);
save(fullfile(cfg.out, 'paired_trials.mat'), 'audit', 'group', 'out');
fprintf('Paired policies: seed %d | seed %d | ordered blocks\n', cfg.seed, cfg.confirmSeed);
for j = 1:np
    fprintf('%-28s %.4f %.4f %.4f\n', out.policy(j), acc(:, j));
end
fprintf('Selected %s, %d workers, %.1f s\n', out.policy(pick), cfg.workers, out.elapsed);
end

function [audit, group, policy] = bench(c, cfg)
group = [ones(1, cfg.outerReps), 2*ones(1, cfg.confirmReps), 3];
for i = 1:numel(c)
    d = c(i).cal;
    n = numel(d.id);
    fold = cfg;
    fold.folds = cfg.outerFolds;
    fold.reps = cfg.outerReps;
    map = mem_folds(d, fold);
    fold.seed = cfg.confirmSeed;
    fold.reps = cfg.confirmReps;
    map = [map, mem_folds(d, fold), floor((0:n-1)' * fold.folds / n) + 1];
    opt = labOptions(cfg, d.ch);
    fprintf('\nPaired case %d: %s\n', i, d.name);
    for j = 1:numel(group)
        for f = 1:fold.folds
            va = map(:, j) == f;
            h = c(i);
            h.cal = mem_take(d, ~va);
            h.qry = mem_take(d, va);
            h.qry.y = [];
            r = mem_labfit(h, opt);
            if j == 1 && f == 1
                a.pred = false(n, 2, numel(r.policy), numel(group));
                a.names = r.names;
                for k = 1:2
                    a.cand{k} = false(n, numel(r.names{k}), numel(group));
                end
            end
            a.pred(va, :, :, j) = r.pred;
            for k = 1:2
                a.cand{k}(va, :, j) = r.cand{k};
            end
        end
        fprintf('  Fold repeat %d/%d completed.\n', j, numel(group));
    end
    a.id = d.id;
    a.y = d.y;
    a.weight = c(i).weight;
    audit(i) = a; %#ok<AGROW>
    clear a
end
policy = r.policy;
end

function opt = labOptions(cfg, ch)
opt = cfg;
opt.reps = cfg.innerReps;
opt.useExtra = ch == 3;
if ch > 1
    opt.behaviorStrength = [0 .45 .70 1];
end
end

function writeReport(out, audit, cfg, group, ok, submit)
[fd, msg] = fopen(out.report, 'w', 'n', 'UTF-8');
assert(fd >= 0, '%s', msg);
clean = onCleanup(@() fclose(fd)); %#ok<NASGU>
fprintf(fd, 'Same outer trial folds: seed %d x %d, seed %d x %d, ordered blocks.\n', ...
    cfg.seed, cfg.outerReps, cfg.confirmSeed, cfg.confirmReps);
fprintf(fd, 'All three local checks use the same original labels.\n');
fprintf(fd, 'Fixed choices came from the pilot runs. Inner policies select inside each outer fold.\n');
fprintf(fd, 'Exported cross-day models use source training and target calibration.\n');
fprintf(fd, 'Policy                       seed1  seed2  blocks  passed  exportable\n');
for j = 1:numel(out.policy)
    fprintf(fd, '%-28s %.4f %.4f %.4f %d %d\n', out.policy(j), out.acc(:, j), ok(j), submit(j));
end
fprintf(fd, '\nSelected %s. Runtime %.1f s using %d workers.\n', ...
    out.policy(out.pick), out.elapsed, cfg.workers);
for i = 1:numel(audit)
    a = audit(i);
    fprintf(fd, '\nGroup %d (weight %.2f, %d labeled trials)\n', i, a.weight, numel(a.id));
    for j = 1:3
        x = out.detail(i, :, 1, j);
        y = out.detail(i, :, out.pick, j);
        gain = mean(double(a.pred(:, :, out.pick, group == j) == a.y) - ...
            double(a.pred(:, :, 1, group == j) == a.y), 4);
        fprintf(fd, 'Split %d, v2 %.4f / %.4f, selected %.4f / %.4f\n', j, x, y);
        fprintf(fd, '  Odor improved %d worsened %d. Behavior improved %d worsened %d.\n', ...
            nnz(gain(:, 1) > 0), nnz(gain(:, 1) < 0), nnz(gain(:, 2) > 0), nnz(gain(:, 2) < 0));
    end
    fprintf(fd, 'Final odor %s, behavior %s\n', out.chosen(i, 1), out.chosen(i, 2));
    for k = 1:2
        fprintf(fd, 'Independent candidates for output %d (seed1 / seed2 / blocks):\n', k);
        for j = 1:numel(a.names{k})
            v = a.cand{k}(:, j, :) == a.y(:, k);
            fprintf(fd, '  %-32s %.4f %.4f %.4f\n', a.names{k}(j), ...
                mean(v(:, :, group == 1), 'all'), mean(v(:, :, group == 2), 'all'), ...
                mean(v(:, :, group == 3), 'all'));
        end
    end
end
end
