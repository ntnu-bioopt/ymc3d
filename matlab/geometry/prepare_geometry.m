function prepare_geometry(filename)
	dx = 20e-3; % discretization (mm)
	dy = 20e-3; % discretization (mm)
	dz = 20e-3; % discretization (mm)

	epi = 0;% epidermis
	papder = 1;% papillary dermis
	retder = 2;% reticulary dermis
	subc = 3;% subcutis
	mole = 4;% mole
	vessel = 5; % blood vessel

	%depths in mm
	length_x = 1; %mm
	length_y = 1;
	length_z = 1.7;

	epiLength = 0.1; %mm
	derLength = 1.5; %mm
	papLength = 0.1*derLength;
	retLength = derLength - papLength;

	%depths in terms of voxels
	epiVox = floor(epiLength/dz);
	papVox = epiVox + floor(papLength/dz);
	retVox = papVox + floor(retLength/dz);

	Nx = length_x/dx;
	Ny = length_y/dy;
	Nz = length_z/dz;
	VOI = zeros(Nx,Ny,Nz);

	VOI(:,:,1:epiVox) = epi;
	VOI(:,:,epiVox+1:papVox) = papder; 
	VOI(:,:,papVox+1:retVox) = retder; 
	VOI(:,:,retVox+1:Nz) = subc;

	% single vessel
	x = length_x/2.0;
	rad = 0.120/2.0; %vessel radius
	d = 0.250; %distance between tissue surface and horizontal vessel axis

	for i = 1:Nx
	    for k = 1:Nz
		for j=1:Ny
			if (((i*dx - x)^2 + (k*dz - d)^2) <= rad^2)
				VOI(i,j,k) = vessel;
			end
		end
	    end
	end


	% write geometry to file
	write_geometry_file(filename, VOI, dx, dy, dz);
end
