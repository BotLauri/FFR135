function xTest2 = LoadMNIST2()

assert(exist(fullfile(pwd, 'xTest2.bin.gz'), 'file')==2,'You must download the file xTest2.bin.gz!')

gunzip('xTest2.bin.gz');
fid = fopen('xTest2.bin');
xTest2 = uint8(fread(fid,'uint8'));
fclose all;
xTest2 = reshape(xTest2,[28 28 1 10000]);

end

