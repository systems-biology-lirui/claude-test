% Compare the regional and population-dynamics candidate on the same folds.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab'));
out = run_memory('regional');
assert(numel(out.pred) == 4 && isfinite(out.score) && isfile(out.zip));
fprintf('REGIONAL_LOCAL_SCORE %.4f\n', out.score);
