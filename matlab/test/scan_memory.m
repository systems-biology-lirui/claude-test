% Compare time windows on labeled memory trials before choosing the pipeline.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab', 'memory'));
cfg.folds = 5;
cfg.reps = 4;
cfg.seed = 42;
cfg.lambda = [0.02 0.12 0.70];
cfg.rbf = [0 0.65];
cfg.relevance = [0 0.75];
cfg.popLambda = [0.01 0.10 1];
cfg.sourceMix = [.25 1 4];
cfg.useProjection = true;
pool = gcp('nocreate');
if isempty(pool)
    pool = parpool('threads');
end
cfg.workers = pool.NumWorkers;
fprintf('Parallel workers: %d\n', cfg.workers);
c = mem_cases(mem_read(fullfile(root, 'player_data.zip')));

for k = 1:numel(c)
    cal = c(k).cal;
    t = tic;
    res = mem_scan(cal, c(k).qry, cfg);
    ok = (res.cv >= .50) == cal.y;
    acc = reshape(mean(mean(ok, 1), 3), 2, []);
    fprintf('\n%s: %d configurations, %.2f s\n', cal.name, size(acc, 2), toc(t));
    for j = 1:2
        [a, order] = sort(acc(j, :), 'descend');
        fprintf('Label %d, all-class baseline %.3f\n', j, max(mean(cal.y(:, j)), 1-mean(cal.y(:, j))));
        for idx = 1:9
            i = order(idx);
            p = res.grid(i);
            fprintf('%.3f %-16s lambda %.2f rbf %.2f rel %.2f\n', ...
                a(idx), res.names(i), p.lam, p.rbf, p.rel);
        end
    end
    res = mem_transfer(c(k), cfg);
    acc = reshape(mean(mean((res.cv >= .50) == cal.y, 1), 3), 2, []);
    for j = 1:2
        [a, order] = sort(acc(j, :), 'descend');
        fprintf('Population label %d:', j);
        for i = 1:min(6, numel(order))
            fprintf(' %.3f %s', a(i), res.names(order(i)));
        end
        fprintf('\n');
    end
end
