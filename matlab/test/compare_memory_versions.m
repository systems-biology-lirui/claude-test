% Compare the two versions on the same alternate seed before exporting v2.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab'));
out = run_memory('compare');
assert(isfinite(out.gain));
