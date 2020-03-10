Function Set-OperationsInfo
{
    @(
        @{Name = "Scan System for installed programs"; ShortName = "SCANPROG"; Info = "Scan System for installed programs";
          Verification = @( @{Function = "VerifyScanPrograms" } );
          Download = @( ) ;
          Action = @() ;
         },
        @{Name = "Libtorch 1.4 Install"; ShortName = "LIBTORCH-1.4"; Info = "Install LIBTORCH-1.3";
          Verification = @( @{Function = "VerifyDirectory"; Path = "$LibTorchPath"; } );
          Download = @( @{Function = "Download"; Source = "https://download.pytorch.org/libtorch/cu101/libtorch-win-shared-with-deps-1.4.0.zip"; Destination = "$localCache\libtorch-win-shared-with-deps-1.3.0.zip" } );
          Action = @( @{Function = "ExtractAllFromZip"; zipFileName = "$localCache\libtorch-win-shared-with-deps-1.3.0.zip"; destinationFolder = "$3rdPartyPath" ; } ) ;
        }
        ,
        @{Name = "OpenCV-4.1.1-vc14-vc15"; ShortName = "CV-411"; Info = "Install OpenCV-4.1.1";
          Verification = @( @{Function = "VerifyDirectory"; Path = "$OpenCVBasePath"; } );
          Download = @( @{Function = "Download"; Source = "https://github.com/opencv/opencv/releases/download/4.1.1/opencv-4.1.1-vc14_vc15.exe"; Destination = "$localCache\opencv-4.1.1-vc14_vc15.exe" } );
          Action = @( @{Function = "InstallExe"; Command = "$localCache\opencv-4.1.1-vc14_vc15.exe"; Param = "-y -o$3rdPartyPath"; Message="Installing opencv ....";}); 
        }    
    )
}
