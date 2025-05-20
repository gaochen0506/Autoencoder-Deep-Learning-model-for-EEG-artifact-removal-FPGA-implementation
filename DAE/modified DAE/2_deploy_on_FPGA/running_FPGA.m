%loading data
load('autoencoder2DNetwork.mat', 'autoencoder');
load('x_test_noisy1.mat', 'data_to_test');
load('x_test_clean1.mat', 'data_to_show');

net = autoencoder;
hTarget = dlhdl.Target("Xilinx",Interface="Ethernet",IPAddress="192.168.1.102");
hdlsetuptoolpath('ToolName', 'Xilinx Vivado', 'ToolPath', 'E:\Xilinx\Vivado\2024.2\bin\vivado.bat');
hW = dlhdl.Workflow(Network=net,Bitstream='zc706_single',Target=hTarget);
%hW.compile('InputFrameNumberLimit', 50);
dn = compile(hW,'InputFrameNumberLimit',1000);
deploy(hW);

%B = 571;

middlenum = zeros(800, 1712, 'single');

totaldata = reshape(data_to_test',[800,1,1,1712]);

firstgroup = totaldata(:,:,:,1:571);
inputImg = dlarray(single(firstgroup),'SSCB');
[prediction,speed] = hW.predict(single(inputImg),'Profile','on');
outputs = extractdata(prediction);
middlenum(:,1:571) = reshape(outputs,[800,571]);

%outputfromH = 1./(1+exp(-reshape(middlenum(:,:),[800,571])));

secondgroup = totaldata(:,:,:,572:1142);
inputImg = dlarray(single(secondgroup),'SSCB');
[prediction,speed] = hW.predict(single(inputImg),'Profile','on');
outputs = extractdata(prediction);
middlenum(:,572:1142) = reshape(outputs,[800,571]);

%outputfromH = 1./(1+exp(-reshape(middlenum(:,:),[800,1712])));

thirdgroup = totaldata(:,:,:,1143:1712);
inputImg = dlarray(single(thirdgroup),'SSCB');
[prediction,speed] = hW.predict(single(inputImg),'Profile','on');
outputs = extractdata(prediction);
middlenum(:,1143:1712) = reshape(outputs,[800,570]);

outputfromH = 1./(1+exp(-reshape(middlenum(:,:),[800,1712])));
%save('outputfromH.mat', 'outputfromH');