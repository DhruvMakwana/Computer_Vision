<img alt="" src="images/image5.png" style="width: 664.00px; height: 388.50px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title="Placeholder image"></span></p>
<p class="c41 title" id="h.2gazcsgmxkub">
   <span class="c35 c40">
<h1>VGG</h1></span></p>
<p class="c47 c51 subtitle" id="h.ng30guuqqp2v"><span class="c35 c23">10 Apr 2015</span></p>
<p class="c17"><span class="c16">&#9472;</span></p>
<p class="c47 c53"><span class="c6"><b>Author</b></span></p>
Karen Simonyan<br>Andrew Zisserman<br>Visual Geometry Group
<p class="c36 c8"><span class="c0"></span></p>
<h1 class="c27" id="h.hur4s6fzpdm2"><span>Overview</span></h1>
<p class="c5"><span>With ConvNets becoming more of a commodity in the computer vision field, a number of attempts have been made to improve the original architecture of Krizhevsky et al. (2012) in a bid to achieve better accuracy. In this paper, they address an important aspect of ConvNet architecture design &ndash; its depth. To this end, authors fix other parameters of the architecture, and steadily increase the depth of the network by adding more convolutional layers, which is feasible due to the use of very small (3 &times; 3) convolution filters in all layers. As a result, they come up with significantly more accurate ConvNet architectures, which not only achieve the state-of-the-art accuracy on ILSVRC classification and localisation tasks, but are also applicable to other image recognition datasets. </span></p>
<h1 class="c27" id="h.3at9u9s4e0vp"><span class="c45 c16 c42">CONVNET CONFIGURATIONS</span></h1>
<p class="c5"><span class="c0">All ConvNet layer configurations are designed using the same principles. First ConvNet Architecture, then configuration and then discussion.</span></p>
<h2 class="c41" id="h.yagilv6ilm3d"><span class="c6">ARCHITECTURE</span></h2>
<p class="c5"><span class="c0">During training, the input to ConvNets is a fixed-size 224 &times; 224 RGB image. The only preprocessing done is subtracting the mean RGB value, computed on the training set, from each pixel. The image is passed through a stack of convolutional (conv.) layers, with filters of a very small receptive field: 3 &times; 3. (Note that In one of the configurations authors also utilise 1 &times; 1 convolution filters, which can be seen as a linear transformation of the input channels). The convolution stride is fixed to 1 pixel. The padding is also 1 pixel for 3 &times; 3 conv. Layers. Pooling is carried out by five max-pooling layers, which follow some of the conv. layers. &nbsp;Max-pooling is performed over a 2 &times; 2 pixel window, with stride 2.</span></p>
<p class="c5"><span class="c0">A stack of convolutional layers is followed by three Fully-Connected (FC) layers: the first two have 4096 channels each, the third contains 1000 channels (one for each class). The final layer is the soft-max layer. </span></p>
<p class="c5"><span class="c0">All hidden layers are equipped with the rectification non-linearity. (Note: None of their Networks(except for one) contain Local Response Normalisation (LRN) normalisation, such normalisation does not improve the performance. </span></p>
<h2 class="c14" id="h.rtm874ipcgnf"><span class="c6"></span></h2>
<h2 class="c14" id="h.sz29t6dyy4tg"><span class="c6"></span></h2>
<h2 class="c41" id="h.p9qyd0p04wzh"><span class="c6">CONFIGURATIONS</span></h2>
<p class="c5"><span class="c0">The ConvNet configurations, evaluated in below Table. Names of the networks are given from A to E. &nbsp;All configurations follow the generic design and differ only in the depth (A: 11 layers; B: 13 layers; C: 16 layers; D: 16 layers; E: 19 layers). </span></p>
<p class="c5"><span class="c0">The width of the network is small starting from 64 and increasing by a factor of 2 until it reaches 512.</span></p>
<p class="c5"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 669.00px; height: 496.50px;"><img alt="" src="images/image4.png" style="width: 669.00px; height: 478.65px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c5"><span class="c0">In the above image VGG-11 is network A, VGG-11 (LRN) is network A-LRN, VGG-13 is network B, VGG-16 (conv1) is network C, VGG-16 is network D, VGG-19 is network E. </span></p>
<ol class="c54 lst-kix_tsmzxtfebi3d-0 start" start="1">
   <li class="c5 c9"><span class="c0">VGG-11 has a 10.4% error rate</span></li>
   <li class="c5 c9"><span class="c0">VGG-11 (LRN) has a 10.5% error rate, is the one with additional local response normalization (LRN) layer. The error rate does not improve, which means LRN is not useful.</span></li>
   <li class="c5 c9"><span class="c0">VGG-13 obtains 9.9% error rate, which means increasing depth helps in accuracy.</span></li>
   <li class="c5 c9"><span class="c0">VGG-16 (Conv1) has a 9.4% error rate, which means the additional three 1&times;1 conv layers help the classification accuracy. 1&times;1 conv actually helps to increase non-linearity of the decision function as stated in paper.</span></li>
   <li class="c5 c9"><span class="c0">VGG-16 has 8.8% error rate, which means increasing depth is still useful.</span></li>
   <li class="c5 c9"><span class="c0">VGG-19 has a 9.0% error rate which means the deep learning network is NOT improving by increasing the depth. Thus, authors stop adding layers.</span></li>
</ol>
<h1 class="c27" id="h.8fq273tenbq8"><span class="c6">Parameters of VGG16(not counting biases)</span></h1>
<p class="c5"><span class="c0">INPUT: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[224x224x3]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0</span></p>
<p class="c5"><span class="c0">CONV3-64:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[224x224x64] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*3)*64 = 1,728</span></p>
<p class="c5"><span class="c0">CONV3-64: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[224x224x64]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*64)*64 = 36,864</span></p>
<p class="c5"><span class="c0">POOL2: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[112x112x64]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 0</span></p>
<p class="c5"><span class="c0">CONV3-128: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[112x112x128] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*64)*128 = 73,728</span></p>
<p class="c5"><span class="c0">CONV3-128: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[112x112x128] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*128)*128 = 147,456</span></p>
<p class="c5"><span class="c0">POOL2: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[56x56x128] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0</span></p>
<p class="c5"><span class="c0">CONV3-256: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[56x56x256] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*128)*256 = 294,912</span></p>
<p class="c5"><span class="c0">CONV3-256: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[56x56x256] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*256)*256 = 589,824</span></p>
<p class="c5"><span class="c0">CONV3-256: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[56x56x256] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*256)*256 = 589,824</span></p>
<p class="c5"><span class="c0">POOL2: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[28x28x256] &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0</span></p>
<p class="c5"><span class="c0">CONV3-512: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[28x28x512]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*256)*512 = 1,179,648</span></p>
<p class="c5"><span class="c0">CONV3-512: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[28x28x512]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*512)*512 = 2,359,296</span></p>
<p class="c5"><span class="c0">CONV3-512: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[28x28x512]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*512)*512 = 2,359,296</span></p>
<p class="c5"><span class="c0">POOL2: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[14x14x512]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0</span></p>
<p class="c5"><span class="c0">CONV3-512: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[14x14x512]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*512)*512 = 2,359,296</span></p>
<p class="c5"><span class="c0">CONV3-512: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[14x14x512]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*512)*512 = 2,359,296</span></p>
<p class="c5"><span class="c0">CONV3-512: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[14x14x512]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(3*3*512)*512 = 2,359,296</span></p>
<p class="c5"><span class="c0">POOL2: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[7x7x512]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;0</span></p>
<p class="c5"><span class="c0">FC: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1x1x4096]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 7*7*512*4096 = 102,760,448</span></p>
<p class="c5"><span class="c0">FC: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1x1x4096]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4096*4096 = 16,777,216</span></p>
<p class="c5"><span class="c0">FC: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;[1x1x1000]&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;params: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4096*1000 = 4,096,000</span></p>
<p class="c5 c48"><span>TOTAL: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;138M parameters</span></p>
<h1 class="c27" id="h.4p7xi5bvhxdr"><span class="c25">Observation</span></h1>
<p class="c5"><span class="c49">What can be gained by using, for instance, a stack of three 3&times;3 conv. layers instead of a single 7&times;7 layer?</span><span>&nbsp;First, if we incorporate three non-linear rectification layers instead of a single one, then it makes the decision function more discriminative. Second, the number of parameters are decreased. The incorporation of 1 &times; 1 conv. layers is a way to increase the nonlinearity of the decision function without affecting the receptive fields of the conv. layers. </span><span class="c0">Thus, large-size filters such as 11&times;11 in AlexNet and 7&times;7 in ZFNet indeed are not needed.</span></p>
<p class="c5"><span class="c0">1 layer of 11&times;11 filter, number of parameters = 11&times;11=121</span></p>
<p class="c5"><span class="c0">5 layer of 3&times;3 filter, number of parameters = 3&times;3&times;5=45</span></p>
<p class="c5"><span class="c0">Number of parameters is reduced by 63%</span></p>
<p class="c5"><span class="c0">1 layer of 7&times;7 filter, number of parameters = 7&times;7=49</span></p>
<p class="c5"><span class="c0">3 layers of 3&times;3 filters, number of parameters = 3&times;3&times;3=27</span></p>
<p class="c5"><span class="c0">Number of parameters is reduced by 45%</span></p>
<p class="c5"><span class="c0">By using 1 layer of 5&times;5 filter, number of parameters = 5&times;5=25</span></p>
<p class="c5"><span class="c0">By using 2 layers of 3&times;3 filters, number of parameters = 3&times;3+3&times;3=18</span></p>
<p class="c5"><span class="c0">Number of parameters is reduced by 28%</span></p>
<p class="c5"><span>With fewer parameters to be learnt, it is better for faster convergence, and reduced overfitting problems.</span></p>
<p class="c5 c8"><span class="c0"></span></p>
<h1 class="c27" id="h.yyrhu7ml5bea"><span class="c45 c16 c42">CLASSIFICATION FRAMEWORK</span></h1>
<h2 class="c41" id="h.buwz1tcz7y35"><span class="c6">TRAINING</span></h2>
<p class="c5"><span>The training is carried out by optimising the multinomial logistic regression using mini-batch gradient descent with momentum. The batch size was set to 256, momentum to 0.9. The training was regularised by weight decay (the L2 penalty multiplier set to 5 &middot; 10&minus;4 ) and dropout regularisation for the first two fully-connected layers (dropout ratio set to 0.5). The learning rate was initially set to 10</span><span class="c44">&minus;2</span><span class="c0">&nbsp;, and then decreased by a factor of 10 when the validation set accuracy stopped improving. In total, the learning rate was decreased 3 times, and the learning was stopped after 370K iterations (74 epochs). </span></p>
<p class="c5"><span>For the initialization authors began with training the configuration network A(11 layers), shallow enough to be trained with random initialisation. Then, when training deeper architectures, they initialised the first four convolutional layers and the last three fully connected layers with the layers of net A (the intermediate layers were initialised randomly). Learning rate is not decreased for the pre-initialised layers, allowing them to change during learning. For random initialisation (where applicable), they sampled the weights from a normal distribution with the zero mean and 10</span><span class="c44">&minus;2</span><span class="c0">&nbsp;variance. The biases were initialised with zero. </span></p>
<p class="c5"><span class="c0">To obtain the fixed-size 224&times;224 ConvNet input images, they were randomly cropped from rescaled training images. To further augment the training set, the crops underwent random horizontal flipping and random RGB colour shift.</span></p>
<p class="c5">
   <span class="c6">
<h2>Training image size</h2></span></p>
<p class="c5"><span>Let S be the smallest side of a training image. In this paper there are two approaches for setting the training scale S. The first is to fix S, which corresponds to single-scale training. In paper experiments, authors evaluated models trained at two fixed scales: S = 256 and S = 384. Given a ConvNet configuration, they first trained the network using S = 256. To speed-up training of the S = 384 network, it was initialised with the weights pre-trained with S = 256, and the learning rate was 10</span><span class="c44">&minus;3</span><span class="c0">. </span></p>
<p class="c5"><span class="c0">The second approach to setting S is multi-scale training, where each training image is individually rescaled by randomly sampling S from a certain range [Smin, Smax] (in paper authors have used Smin = 256 and Smax = 512). Since objects in images can be of different size, it is beneficial to take this into account during training. For speed reasons, training of multi-scale models is done by fine-tuning all layers of a single-scale model with the same configuration, pre-trained with fixed S = 384.</span></p>
<p class="c5 c8"><span class="c0"></span></p>
<p class="c5">
   <span class="c6">
<h2>TESTING</h2></span></p>
<p class="c19"><span class="c0">At test time, given a trained ConvNet and an input image. First, it is isotropically rescaled to a pre-defined smallest image side, denoted as test scale Q. (Note that Q is not necessarily equal to the training scale S). Then, the network is applied densely over the rescaled test image. The fully-connected layers are first converted to convolutional layers (the first FC layer to a 7 &times; 7 conv. layer, the last two FC layers to 1 &times; 1 conv. layers). The resulting fully-convolutional net is then applied to the whole (uncropped) image. The result is a class score map with the number of channels equal to the number of classes. Augmentation of the test set is done by horizontal flipping of the images; the soft-max class posteriors of the original and flipped images are averaged to obtain the final scores for the image. </span></p>
<p class="c19"><span class="c0">At the same time, using a large set of crops can lead to improved accuracy, as it results in a finer sampling of the input image compared to the fully-convolutional net. For reference authors &nbsp;also evaluate networks using 50 crops per scale (5 &times; 5 regular grid with 2 flips), for a total of 150 crops over 3 scales, which is comparable to 144 crops over 4 scales.</span></p>
<p class="c19"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 202.67px;"><img alt="" src="images/image2.png" style="width: 624.00px; height: 202.67px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c19"><span class="c0">The first FC is replaced by 7&times;7 conv.</span></p>
<p class="c19"><span class="c0">The second and third FC are replaced by 1&times;1 conv.</span></p>
<p class="c19"><span class="c0">Thus, all FC layers are replaced by conv layers.</span></p>
<p class="c19"><span class="c0">During testing, in VGGNet, the test image is directly go through the VGGNet and obtain a class score map. This class score map is spatially averaged to be a fixed-size vector.</span></p>
<p class="c19 c8"><span class="c0"></span></p>
<p class="c19"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 607.00px; height: 77.00px;"><img alt="" src="images/image3.png" style="width: 607.00px; height: 77.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c19"><span class="c0">By average both dense and multi-crop results, VGG-16 and VGG-19 error rates are reduced to 7.2% and 7.1%.</span></p>
<p class="c19 c8"><span class="c0"></span></p>
<p class="c5">
   <span class="c6">
<h2>IMPLEMENTATION DETAILS</h2></span></p>
<p class="c5"><span class="c0">Implementation is derived from the publicly available C++ Caffe toolbox, but contains a number of significant modifications, to perform training and evaluation on multiple GPUs, as well as train and evaluate on full-size (uncropped) images at multiple scales. Multi-GPU training exploits data parallelism, and is carried out by splitting each batch of training images into several GPU batches, processed in parallel on each GPU. After the GPU batch gradients are computed, they are averaged to obtain the gradient of the full batch. Gradient computation is synchronous across the GPUs, so the result is exactly the same as when training on a single GPU. This conceptually much simpler scheme already provides a speedup of 3.75 times on an off-the-shelf 4-GPU system, as compared to using a single GPU. On a system equipped with four NVIDIA Titan Black GPUs, training a single net took 2&ndash;3 weeks depending on the architecture.</span></p>
<h1 class="c27" id="h.p5kt33sm1ewd"><span class="c45 c16 c42">CLASSIFICATION EXPERIMENTS</span></h1>
<p class="c5">
   <span class="c6">
<h2>DATASET</h2></span></p>
<p class="c5"><span class="c0">The dataset used for this architecture was the ILSVRC-2012 dataset. &nbsp;The dataset includes images of 1000 classes, and is split into three sets: training (1.3M images), validation (50K images), and testing (100K images with held-out class labels). </span></p>
<p class="c5">
   <span class="c6">
<h2>SINGLE SCALE EVALUATION</h2></span></p>
<p class="c5"><span class="c0">Here the performance of individual ConvNet models at a single scale with the layer configurations. The test image size was set as follows: Q = S for fixed S, and Q = 0.5 (Smin + Smax) for jittered S &isin; [Smin, Smax]. </span></p>
<p class="c5"><span class="c0">Classification error decreases with the increased ConvNet depth: from 11 layers in A to 19 layers in E. In spite of the same depth, the configuration C &nbsp;which contains three 1 &times; 1 conv. layers, performs worse than the configuration D, which uses 3 &times; 3 conv. layers throughout the network. The error rate of the architecture saturates when the depth reaches 19 layers, but even deeper models might be beneficial for larger datasets.</span></p>
<p class="c5"><span class="c0">Authors have also compared the net B with a shallow net with five 5 &times; 5 conv. layers, which was derived from B by replacing each pair of 3 &times; 3 conv. layers with a single 5 &times; 5 conv. layer. The top-1 error of the shallow net was measured to be 7% higher than that of B, which confirms that a deep net with small filters outperforms a shallow net with larger filters.</span></p>
<p class="c5"><span class="c0">Scale jittering at training time (S &isin; [256; 512]) leads to significantly better results than training on images with fixed smallest side (S = 256 or S = 384), even though a single scale is used at test time. This confirms that training set augmentation by scale jittering is indeed helpful for capturing multi-scale image statistics.</span></p>
<p class="c5 c8"><span class="c0"></span></p>
<a id="t.584dd60514c70b740f342a851f5550671e86b9d6"></a><a id="t.0"></a>
<table class="c26">
   <tbody>
      <tr class="c1">
         <td class="c2" colspan="1" rowspan="2">
            <p class="c3"><span class="c12">ConvNet config.</span></p>
         </td>
         <td class="c29" colspan="2" rowspan="1">
            <p class="c3"><span class="c12">Smallest image side</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="2">
            <p class="c3"><span class="c12">top-1 val. error (%)</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="2">
            <p class="c3"><span class="c12">top-5 val. error (%)</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c12">Train (S)</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c12">Test (Q)</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">A</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">29.6</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">10.4</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">A - LRN</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">29.7</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">10.5</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">B</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">28.7</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">9.9</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c2" colspan="1" rowspan="3">
            <p class="c3"><span class="c0">C</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">28.1</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">9.4</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">28.1</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">9.3</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">[256;512]</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">27.3</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.8</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c2" colspan="1" rowspan="3">
            <p class="c3"><span class="c0">D</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">27.0</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.8</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">26.8</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.7</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">[256;512]</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">25.6</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.1</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c2" colspan="1" rowspan="3">
            <p class="c3"><span class="c0">E</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">27.3</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">9.0</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">26.9</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.7</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">[256;512]</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">25.5</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.0</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c5"><span class="c0">VGG-13 reduced the error rate from 9.4%/9.3% to 8.8%.</span></p>
<p class="c5"><span class="c0">VGG-16 reduced the error rate from 8.8%/8.7% to 8.1%.</span></p>
<p class="c5"><span>VGG-19 reduced the error rate from 9.0%/8.7% to 8.0%.</span></p>
<p class="c5">
   <span class="c6">
<h2>MULTI-SCALE EVALUATION</h2></span></p>
<p class="c5"><span class="c0">Here we have the effect of scale jittering at test time. It consists of running a model over several rescaled versions of a test image. Considering that a large discrepancy between training and testing scales leads to a drop in performance, the models trained with fixed S were evaluated over three test image sizes, close to the training one: Q = {S &minus; 32, S, S + 32}. At the same time, scale jittering at training time allows the network to be applied to a wider range of scales at test time, so the model trained with variable S &isin; [Smin; Smax] was evaluated over a larger range of sizes Q = {Smin, 0.5(Smin + Smax), Smax}. The results, presented in Table below, indicate that scale jittering at test time leads to better performance. Best single-network performance on the validation set is 24.8%/7.5% top-1/top-5 error. On the test set, the configuration E achieves 7.3% top-5 error.</span></p>
<p class="c5 c8"><span class="c0"></span></p>
<a id="t.1133cfa4a48042a2801f41258d0d34eb159a6bcb"></a><a id="t.1"></a>
<table class="c26">
   <tbody>
      <tr class="c1">
         <td class="c2" colspan="1" rowspan="2">
            <p class="c3"><span class="c12">ConvNet config.</span></p>
         </td>
         <td class="c29" colspan="2" rowspan="1">
            <p class="c3"><span class="c12">Smallest image side</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="2">
            <p class="c3"><span class="c12">top-1 val. error (%)</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="2">
            <p class="c3"><span class="c12">top-5 val. error (%)</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c12">Train (S)</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c12">Test (Q)</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">B</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">224, 256, 288</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">28.2</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">9.6</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c2" colspan="1" rowspan="3">
            <p class="c3"><span class="c0">C</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">224, 256, 288 </span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">27.7</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">9.2</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">352, 384, 416</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">27.8</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">9.2</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">[256;512]</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256, 384, 512</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">26.3</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.2</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c2" colspan="1" rowspan="3">
            <p class="c3"><span class="c0">D</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">224, 256, 288 </span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">26.6</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.6</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">352, 384, 416</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">26.5</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.6</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">[256;512]</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256, 384, 512</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">24.8</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">7.5</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c2" colspan="1" rowspan="3">
            <p class="c3"><span class="c0">E</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">224, 256, 288 </span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">26.9</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.7</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">384</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">352, 384, 416</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">26.7</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">8.6</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">[256;512]</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">256, 384, 512</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">24.8</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">7.5</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c5 c8"><span class="c0"></span></p>
<p class="c5"><span class="c0">VGG-13 reduced the error rate from 9.2%/9.2% to 8.2%,</span></p>
<p class="c5"><span class="c0">VGG-16 reduced the error rate from 8.6%/8.6% to 7.5%,</span></p>
<p class="c5"><span class="c0">VGG-19 reduced the error rate from 8.7%/8.6% to 7.5%,</span></p>
<p class="c5 c8"><span class="c0"></span></p>
<p class="c5">
   <span class="c6">
<h2>MULTI-CROP EVALUATION</h2></span></p>
<p class="c5"><span class="c0">Using multiple crops performs slightly better than dense evaluation, and the two approaches are indeed complementary, as their combination outperforms each of them.</span></p>
<p class="c5"><span class="c0">Comparison between dense ConvNet evaluation and multi-crop evaluation is given in the table below.</span></p>
<p class="c5 c8"><span class="c0"></span></p>
<a id="t.ce6281bdd2b8253e848ef91a3cde856fe30573a5"></a><a id="t.2"></a>
<table class="c26">
   <tbody>
      <tr class="c1">
         <td class="c10" colspan="1" rowspan="2">
            <p class="c3"><span class="c12">ConvNet config.</span></p>
         </td>
         <td class="c13" colspan="2" rowspan="2">
            <p class="c3"><span class="c12">Evaluation method</span></p>
         </td>
         <td class="c43" colspan="1" rowspan="2">
            <p class="c3"><span class="c12">top-1 val. error (%)</span></p>
         </td>
         <td class="c38" colspan="1" rowspan="2">
            <p class="c3"><span class="c12">top-5 val. error (%)</span></p>
         </td>
      </tr>
      <tr class="c1"></tr>
      <tr class="c1">
         <td class="c10" colspan="1" rowspan="3">
            <p class="c3"><span class="c0">D</span></p>
         </td>
         <td class="c46" colspan="2" rowspan="1">
            <p class="c3"><span class="c0">dense</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">24.8</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">7.5</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c13" colspan="2" rowspan="1">
            <p class="c3"><span class="c0">multi-crop</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">24.6</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">7.5</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c13" colspan="2" rowspan="1">
            <p class="c3"><span class="c0">multi-crop &amp; dense</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">24.4</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">7.2</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c10" colspan="1" rowspan="3">
            <p class="c3"><span class="c0">E</span></p>
         </td>
         <td class="c13" colspan="2" rowspan="1">
            <p class="c3"><span class="c0">dense</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">24.8</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">7.5</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c13" colspan="2" rowspan="1">
            <p class="c3"><span class="c0">multi-crop</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">24.6</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">7.4</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c13" colspan="2" rowspan="1">
            <p class="c3"><span class="c0">Multi-crop &amp; dense</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">24.4</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">7.1</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c5 c8"><span class="c0"></span></p>
<p class="c5">
   <span class="c6">
<h2>COMPARISON WITH THE STATE OF THE ART</h2> </span></p>
<p class="c5"><span>In the classification task of the ILSVRC-2014 challenge, &nbsp;&ldquo;VGG&rdquo; team secured 2nd place with</span><span class="c25 c42">&nbsp;</span><span>7.3% test error using an ensemble of 7 models. After the submission, they decreased the error rate to 6.8% using an ensemble of 2 models</span><span class="c6">.</span></p>
<p class="c5 c8"><span class="c6"></span></p>
<a id="t.78af8f7e4b24139a6dccc466624fba0355875610"></a><a id="t.3"></a>
<table class="c26">
   <tbody>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c12">Method</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c12">top-1 val. error (%)</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c12">top-5 val. error (%)</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c12">top-5 test error (%)</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">VGG (2 nets, multi-crop &amp; dense eval.)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">23.7</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">6.8</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">6.8</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">VGG (1 net, multi-crop &amp; dense eval.)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">24.4</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">7.1</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">7.0</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">VGG (ILSVRC submission, 7 nets, dense eval.)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">24.7</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">7.5</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">7.3</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">GoogLeNet (Szegedy et al., 2014) (1 net)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
         <td class="c28" colspan="2" rowspan="1">
            <p class="c15"><span class="c0">7.9</span></p>
         </td>
      </tr>
      <tr class="c1">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">GoogLeNet (Szegedy et al., 2014) (7 nets)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
         <td class="c28" colspan="2" rowspan="1">
            <p class="c15"><span class="c0">6.7</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">MSRA (He et al., 2014) (11 nets) </span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">8.1</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">MSRA (He et al., 2014) (1 net) </span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">27.9</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">9.1</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">9.1</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">Clarifai (Russakovsky et al., 2014) (multiple nets)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">11.7</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">Clarifai (Russakovsky et al., 2014) (1 net) </span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">12.5</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">Zeiler &amp; Fergus (Zeiler &amp; Fergus, 2013) (6 nets)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">36.0</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">14.7</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">14.8</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">Zeiler &amp; Fergus (Zeiler &amp; Fergus, 2013) (1 net)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">37.5</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">16.0</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">16.1</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">OverFeat (Sermanet et al., 2014) (7 nets)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">34.0</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">13.2</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">13.6</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">OverFeat (Sermanet et al., 2014) (1 net) </span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">35.7</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">14.2</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">Krizhevsky et al. (Krizhevsky et al., 2012) (5 nets)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">38.1</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">16.4</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">16.4</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c11" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">Krizhevsky et al. (Krizhevsky et al., 2012) (1 net)</span></p>
         </td>
         <td class="c20" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">40.7</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">18.2</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">-</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c5">
   <span class="c16 c55 c42">
<h1>Localization Task</h1></span></p>
<p class="c5"><span class="c0">For the localization task, a bounding box is represented by a 4-D vector storing its center coordinates, width, and height. Thus, the logistic regression objective is replaced with a Euclidean loss, which penalises the deviation of the predicted bounding box parameters from the ground-truth.</span></p>
<p class="c5"><span>There is a choice of whether the bounding box prediction is shared across all classes (single-class regression, SCR) or is class-specific (per-class regression, PCR). In the former case, the last layer is 4-D, while in the latter it is 4000-D (since there are 1000 classes in the dataset).</span></p>
<p class="c5 c8"><span class="c0"></span></p>
<a id="t.03f658bc25e74bb7f768d06a297fe015670c2950"></a><a id="t.4"></a>
<table class="c26">
   <tbody>
      <tr class="c7">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">Fine-tuned layers</span></p>
         </td>
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">Regression type</span></p>
         </td>
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">GT class localisation error</span></p>
         </td>
      </tr>
      <tr class="c39">
         <td class="c22" colspan="1" rowspan="2">
            <p class="c15"><span class="c0">1st and 2nd FC</span></p>
         </td>
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">SCR</span></p>
         </td>
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">36.4</span></p>
         </td>
      </tr>
      <tr class="c39">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">PCR</span></p>
         </td>
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">34.3</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">all</span></p>
         </td>
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">PCR</span></p>
         </td>
         <td class="c22" colspan="1" rowspan="1">
            <p class="c15"><span class="c0">33.1</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c5"><span class="c0">As shown above, PCR is better than SCR. And fine-tuning all layers is better than just fine-tuned the 1st and 2nd FC layers. The results above are obtained by using just center crop.</span></p>
<p class="c5"><span class="c0">VGGNet even outperforms GoogLeNet and won the localization task in ILSVRC 2014.</span></p>
<p class="c5 c8"><span class="c0"></span></p>
<a id="t.40a7789af183595b3c434572905c2041ce55feef"></a><a id="t.5"></a>
<table class="c32">
   <tbody>
      <tr class="c7">
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c12">Method</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c12">VOC-2007(mean AP)</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c12">VOC-2012(mean-AP)</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c12">Caltech-101</span></p>
            <p class="c3"><span class="c12">(mean class recall)</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c12">Caltec(mean class recall)h-256</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">Zeiler &amp; Fergus </span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">-</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">79.0</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">86.5 &plusmn; 0.5</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">74.2 &plusmn; 0.3</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">Chatfield et al.</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">82.4</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">83.2</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">88.4 &plusmn; 0.6</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">77.6 &plusmn; 0.1</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">He et al.</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">82.4</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">-</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">93.4 &plusmn; 0.5</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">-</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">Wei et al.</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">81.5 </span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">81.7</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">-</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">-</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">VGG Net-D </span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">89.3</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">89.0</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">91.8 &plusmn; 1.0</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">85.0 &plusmn; 0.2</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">VGG Net-E</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">89.3</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">89.0</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">92.3 &plusmn; 0.5</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">85.1 &plusmn; 0.3</span></p>
         </td>
      </tr>
      <tr class="c7">
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">VGG Net-D &amp; Net-E</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">89.7</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">89.3</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">92.7 &plusmn; 0.5</span></p>
         </td>
         <td class="c2" colspan="1" rowspan="1">
            <p class="c3"><span class="c0">86.2 &plusmn; 0.3</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c5"><span class="c0">VGGNet has the best results on VOC 2007, 2012 and Caltech 256 dataset. And it also has competitive results on the Caltech 101 dataset.</span></p>
<p class="c5 c8"><span class="c0"></span></p>
<p class="c5">
   <span class="c16 c42 c55">
<h1>References</h1></span></p>
<p class="c5"><span class="c58"><a class="c34" href="https://www.google.com/url?q=https://arxiv.org/pdf/1409.1556.pdf&amp;sa=D&amp;ust=1591895161802000">VERY DEEP CONVOLUTIONAL NETWORKS FOR LARGE-SCALE IMAGE RECOGNITION</a></span></p>
</body></html>

For Detail report of VGGNet architecture click [here](https://github.com/DhruvMakwana/Computer_Vision/raw/master/VGG/VGG_Detail_Report.pdf)

To implement VGGNet 16 layer architecture in keras using python script run `VGG16_Keras.py` or to implement VGGNet 16 layer architecture in keras using notebook run `VGG16_Keras.ipynb`

To implement VGGNet 19 layer architecture in keras using python script run `VGG19_Keras.py` or to implement VGGNet 19 layer architecture in keras using notebook run `VGG19_Keras.ipynb`

To implement VGGNet 16 layer architecture in pytorch using python script run `VGG16_pytorch.py` or to implement VGGNet 16 layer architecture in pytorch using notebook run `VGG16_pytorch.ipynb`

To implement VGGNet 19 layer architecture in pytorch using python script run `VGG19_pytorch.py` or to implement VGGNet 19 layer architecture in pytorch using notebook run `VGG19_pytorch.ipynb`

Note: We have used Batch Normalization in keras version instead of LRN and used LRN in pytorch version.