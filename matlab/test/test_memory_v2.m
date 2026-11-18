% Regenerate the v2 package and verify it against the official trial IDs.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab'));
out = run_memory('v2');
assert(numel(out.pred) == 4 && isfinite(out.score));
memoryZipToCheck = out.zip;
run(fullfile(root, 'matlab', 'test', 'verify_memory_zip.m'));
clear memoryZipToCheck
