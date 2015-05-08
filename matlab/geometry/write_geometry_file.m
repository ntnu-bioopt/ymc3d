function write_geometry_file(fName, geomMtrx, dx, dy, dz)
%INPUT:
%   fName - file name
%   geomMtrx - geometry matrix [Nz,Ny,Nx]
%   dx - voxel size in x-direction [mm]
%   dy - voxel size in y-direction [mm]
%   dz - voxel size in z-direction [mm]

Nx = size(geomMtrx,1);
Ny = size(geomMtrx,2);
Nz = size(geomMtrx,3);

% Write geometry data
fid =fopen([fName,'.bin'],'wb');   
    
fwrite(fid,3,'integer*4');% 3D file                
fwrite(fid,[Nx,Ny,Nz],'integer*4');%num. of voxels
fwrite(fid,[dx,dy,dz],'double');%voxel size in mm
fwrite(fid,geomMtrx,'integer*4');    

fclose(fid);
end
