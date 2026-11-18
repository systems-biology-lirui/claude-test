% Export the source-trained v3 from the already completed paired experiment.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
addpath(fullfile(root, 'matlab'));
out = run_memory('v3_finish');
assert(abs(out.base - .7449) < .0001);
assert(out.policy(out.pick) == "matched_source_and_behavior" && isfile(out.zip));
assert(all(out.acc(:, out.pick) > out.acc(:, 1)));
memoryZipToCheck = out.zip;
run(fullfile(root, 'matlab', 'test', 'verify_memory_zip.m'));
clear memoryZipToCheck
