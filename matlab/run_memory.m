function out = run_memory(mode)
% 记忆赛道入口。选项集中放在这里，模型只读延迟期放电。
if nargin == 0
    mode = 'base';
end
assert(ismember(mode, {'base','regional','v2','compare','v3','v3_finish'}), 'Unknown memory pipeline mode.');
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(fullfile(here, 'memory'));

cfg.team = '谁说bio不如com';
cfg.pack = fullfile(root, 'player_data.zip');
cfg.out = fullfile(root, 'output', 'memory');
cfg.mode = mode;

% 五折试次划分，最终拟合前重复筛选。外层分数用于检查泛化。
cfg.seed = 42;
cfg.folds = 5;
cfg.reps = 8;
cfg.outerFolds = 5;
cfg.outerReps = 3;
cfg.innerReps = 2;

% 单神经元模型和跨场次脑区模型的候选参数。
cfg.lambda = [.02 .12 .70];
cfg.rbf = [0 .65];
cfg.relevance = [0 .75];
cfg.popLambda = [.01 .10 1];
cfg.sourceMix = [.25 1 4];
cfg.srcWeights = [.10 .25 .45];
cfg.useProjection = true;
cfg.bestViews = 4;
cfg.blendTemp = .035;
cfg.behaviorGain = 1;
cfg.behaviorStrength = [.20 .45 .70 1];
cfg.useExtra = false;
if strcmp(mode, 'regional')
    cfg.out = fullfile(root, 'output', 'memory_regional');
elseif strcmp(mode, 'v2')
    cfg.out = fullfile(root, 'output', 'memory_v2');
elseif ismember(mode, {'v3','v3_finish'})
    cfg.out = fullfile(root, 'output', 'memory_v3');
end

% v3 的配对验证和接纳阈值。两组随机划分及一次按试次顺序分块。
cfg.confirmSeed = 927;
cfg.confirmReps = 3;
cfg.minInnerGain = 1;
cfg.minPackageGain = .002;
cfg.beh.seed = 42;

% 使用线程池共享特征。当前 MATLAB 会话可以复用这个池。
cfg.workers = 0;
if license('test', 'Distrib_Computing_Toolbox')
    try
        pool = gcp('nocreate');
        if isempty(pool)
            pool = parpool('threads');
        end
        cfg.workers = pool.NumWorkers;
    catch err
        warning('Thread pool unavailable, using MATLAB BLAS: %s', err.message);
    end
end
fprintf('Memory pipeline: %d parallel workers, up to %d BLAS threads\n', cfg.workers, maxNumCompThreads);
t = tic;
c = mem_cases(mem_read(cfg.pack));
if ismember(mode, {'v3','v3_finish'})
    out = mem_lab(c, cfg);
    return
end
if strcmp(mode, 'compare')
    out = compare(c, cfg);
    return
end
for j = 1:numel(c)
    fprintf('\nChallenge %d  %s\n', c(j).qry.ch, c(j).qry.name);
    opt = variant(cfg, mode, c(j).qry.ch);
    v(j) = mem_validate(c(j), opt); %#ok<AGROW>
    [y, f] = mem_predict(c(j), opt);
    p(j) = struct('id', c(j).qry.id, 'y', y, 'name', c(j).qry.name); %#ok<AGROW>
    info(j) = f; %#ok<AGROW>
    fprintf('Outer validation: cue %.4f, behavior %.4f; mean-rate baseline %.4f / %.4f\n', ...
        v(j).acc, v(j).base);
    fprintf('Classified %d trials with %d candidate configurations.\n', numel(p(j).id), f.count);
end

out.zip = mem_pack(p, cfg);
out.pred = p;
out.val = v;
out.workers = cfg.workers;
out.elapsed = toc(t);
out.score = sum([c.weight] .* arrayfun(@(a) mean(a.acc), v));
out.base = sum([c.weight] .* arrayfun(@(a) mean(a.base), v));
out.report = fullfile(cfg.out, 'validation.txt');
writeReport(out, c, info, cfg);
fprintf('Local weighted accuracy %.4f, mean-rate baseline %.4f\n', out.score, out.base);
fprintf('Built ZIP in %.1f s: %s\n', out.elapsed, out.zip);
end

function cfg = variant(cfg, mode, ch)
% 跨个体使用脑区特征。跨天行为模型在证据不足时缩回训练集先验。
cfg.useExtra = strcmp(mode, 'regional') || (strcmp(mode, 'v2') && ch == 3);
if cfg.useExtra || (strcmp(mode, 'v2') && ch == 2)
    cfg.behaviorStrength = [0 .45 .70 1];
end
end

function out = compare(c, cfg)
% Compare the two versions using a different, fixed trial split.
cfg.seed = 211;
out.gain = 0;
for j = 2:numel(c)
    a = mem_validate(c(j), cfg);
    b = mem_validate(c(j), variant(cfg, 'v2', c(j).qry.ch));
    out.gain = out.gain + c(j).weight * mean(b.acc - a.acc);
    fprintf('New split %s: v1 %.4f / %.4f, v2 %.4f / %.4f\n', ...
        c(j).cal.name, a.acc, b.acc);
end
fprintf('New split: weighted v2 minus v1 gain = %.4f\n', out.gain);
end

function writeReport(out, c, info, cfg)
[fd, msg] = fopen(out.report, 'w', 'n', 'UTF-8');
assert(fd >= 0, '%s', msg);
clean = onCleanup(@() fclose(fd)); %#ok<NASGU>
fprintf(fd, 'Memory track %s: local nested validation.\n', cfg.mode);
fprintf(fd, '%d repetitions of %d outer folds. Inner selection uses %d repeats.\n', ...
    cfg.outerReps, cfg.outerFolds, cfg.innerReps);
fprintf(fd, 'Cross-session validation uses held-out eval-1 calibration trials.\n');
fprintf(fd, '%d workers; elapsed %.1f s.\n\n', out.workers, out.elapsed);
for j = 1:numel(c)
    v = out.val(j);
    a = c(j);
    fprintf(fd, '%s\n', a.qry.name);
    fprintf(fd, 'Labeled %d / predicted %d / official weight %.2f\n', ...
        numel(a.cal.id), numel(a.qry.id), a.weight);
    fprintf(fd, 'Cue / behavior accuracy: %.4f / %.4f\n', v.acc);
    fprintf(fd, 'Cue / behavior balanced accuracy: %.4f / %.4f\n', v.bal);
    fprintf(fd, 'Mean-rate baseline: %.4f / %.4f\n', v.base);
    fprintf(fd, 'Candidate models: %d\n', info(j).count);
    fprintf(fd, 'Cue views: %s\n', strjoin(info(j).names{1}, ', '));
    fprintf(fd, 'Behavior views: %s\n', strjoin(info(j).names{2}, ', '));
    fprintf(fd, 'Cross-session model weights: %.2f / %.2f\n', info(j).srcWeight);
    fprintf(fd, 'Behavior calibration strength: %.2f\n\n', info(j).behaviorStrength);
end
fprintf(fd, 'Local weighted accuracy: %.4f. Mean-rate baseline: %.4f.\n', out.score, out.base);
end
