# This package imlements optimized I/O access for Torch DL applications

 There are two parts  
   
 1) Installing the package
	
	just have to type "luarocks make" in the torchIO folder.

 2) Creating the LMDB. 

From the torchIO/createLMDB folder, type  ./create_lmdb path_to_imagenet  path_to_write_lmdb

where path_to_imagenet is the location of imagenet directory ( there should be train/ and val/ folders inside this dir) and /path_to_write_lmdb is where you have to write your lmdb files. 

If both locations are in SSD, this should take about 30 mins and the size of the created LMDBs should be around 440GB.

Please ensure the imagnet is resized using "find . -name "*.JPEG" | xargs -I {} convert {} -resize "256^>" {}"
before creating the LMDB database