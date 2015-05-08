function write_optical_props_file(fname, optical_data)
%input: 
% fname: filename
% optical_data: a num_tissue_types x 4 data matrix. Cols are arrayed with mua, mus, g, n. 
	[num_tissues, num_optprops] = size(optical_data);
	fid = fopen([fname, 'bid'], 'wb');
	fwrite(fid, num_tissues, 'integer*4');
	fwrite(fid, optical_data, 'double');
	fclose(fid);
end
