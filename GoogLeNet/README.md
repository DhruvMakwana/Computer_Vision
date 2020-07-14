<img alt="" src="images/image1.png" style="width: 621.14px; height: 8.14px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title="horizontal line"></span><span class="c47">&nbsp;</span></h2>
<p class="c8"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 352.00px;"><img alt="" src="images/image6.png" style="width: 624.00px; height: 352.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c2 title" id="h.2gazcsgmxkub"><span class="c12"><h1>GoogLeNet</h1></span></p>
<p class="c27 subtitle" id="h.ng30guuqqp2v"><span class="c9">17.09.2014</span></p>
<p class="c48"><span class="c40 c46">&#9472;</span></p>
<p class="c34"><span class="c29">Author</span></p>
Christian Szegedy&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dragomir Anguelov<br>
Wei Liu&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Dumitru Erhan<br>
Yangqing Jia&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Vincent Vanhoucke<br>
Pierre Sermanet&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Andrew Rabinovich<br>
Scott Reed
<p class="c38 c26"><span class="c9"></span></p>
<h1 class="c19" id="h.au51mny0sx6"><span>Overview</span></h1>
<p class="c8"><span class="c1">GoogLeNet submission to ILSVRC 2014 actually uses 12&times; fewer parameters than the winning architecture of AlexNet from two years ago, while being significantly more accurate. For most of the experiments, the models were designed to keep a computational budget of 1.5 billion multiply-adds at inference time, so that the they do not end up to be a purely academic curiosity, but could be put to real world use, even on large datasets, at a reasonable cost. The name Inception derived from Network in network paper and famous &ldquo;we need to go deeper&rdquo; internet meme. </span></p>
<p class="c25"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 400.00px; height: 226.00px;"><img alt="" src="images/image7.png" style="width: 400.00px; height: 226.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<h1 class="c19" id="h.3at9u9s4e0vp"><span class="c32">Related Work</span></h1>
<p class="c8"><span class="c1">Starting with LeNet-5, convolutional neural networks (CNN) have typically had a standard structure &ndash; stacked convolutional layers (optionally followed by contrast normalization and max pooling) are followed by one or more fully-connected layers. Despite concerns that max-pooling layers result in loss of accurate spatial information, the same convolutional network architecture has also been successfully employed for localization, object detection and human pose estimation. Furthermore, Inception layers are repeated many times, leading to a 22-layer deep model in the case of the GoogLeNet model. </span></p>
<p class="c8"><span>Network-in-Network contains 1&times;1 convolutional layers. Here 1 &times; 1 convolutions have dual purpose: most critically, they are used mainly as dimension reduction modules to remove computational bottlenecks that would otherwise limit the size of our networks.&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;</span></p>
<h1 class="c19" id="h.4p7xi5bvhxdr"><span>Motivation and High Level Considerations</span></h1>
<p class="c8"><span class="c1">The most straightforward way of improving the performance of deep neural networks is by increasing their size. However this simple solution comes with two major drawbacks. Bigger size typically means a larger number of parameters, which makes the enlarged network more prone to overfitting. Another drawback of uniformly increased network size is the dramatically increased use of computational resources. if two convolutional layers are chained, any uniform increase in the number of their filters results in a quadratic increase of computation.</span></p>
<p class="c8"><span class="c1">The fundamental way of solving both issues would be by ultimately moving from fully connected to sparsely connected architectures, even inside the convolutions. &nbsp;If the probability distribution of the data-set is representable by a large, very sparse deep neural network, then the optimal network topology can be constructed layer by layer by analyzing the correlation statistics of the activations of the last layer and clustering neurons with highly correlated outputs. Although the strict mathematical proof requires very strong conditions, the fact that this statement resonates with the well known Hebbian principle &ndash; neurons that fire together, wire together &ndash; suggests that the underlying idea is applicable even under less strict conditions, in practice. </span></p>
<p class="c8"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 206.00px; height: 310.50px;"><img alt="" src="images/image4.png" style="width: 206.00px; height: 310.50px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 321.50px; height: 343.00px;"><img alt="" src="images/image9.png" style="width: 321.50px; height: 343.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c8"><span>&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; Dense Architecture&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;Sparse Architecture</span></p>
<h1 class="c19" id="h.yyrhu7ml5bea"><span class="c32">Architectural Details</span></h1>
<p class="c8"><span class="c1">The main idea of the Inception architecture is based on finding out how an optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components. All we need is to find the optimal local construction and to repeat it spatially. &nbsp;Arora et al. suggests a layer-by layer construction in which one should analyze the correlation statistics of the last layer and cluster them into groups of units with high correlation. One can also expect that there will be a smaller number of more spatially spread out clusters that can be covered by convolutions over larger patches, and there will be a decreasing number of patches over larger and larger regions. In order to avoid patch alignment issues, current incarnations of the Inception architecture are restricted to filter sizes 1&times;1, 3&times;3 and 5&times;5.</span></p>
<p class="c8"><span class="c1">As these &ldquo;Inception modules&rdquo; are stacked on top of each other, their output correlation statistics are bound to vary: as features of higher abstraction are captured by higher layers, their spatial concentration is expected to decrease suggesting that the ratio of 3&times;3 and 5&times;5 convolutions should increase as we move to higher layers. </span></p>
<p class="c8"><span class="c1">One big problem with the above modules, at least in this na&uml;&#305;ve form, is that even a modest number of 5&times;5 convolutions can be prohibitively expensive on top of a convolutional layer with a large number of filters. This problem becomes even more pronounced once pooling units are added to the mix. This leads to the second idea of the proposed architecture: judiciously applying dimension reductions and projections wherever the computational requirements would increase too much otherwise. </span></p>
<p class="c25"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 427.50px; height: 335.38px;"><img alt="" src="images/image11.png" style="width: 427.50px; height: 335.38px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c25"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 554.00px; height: 312.00px;"><img alt="" src="images/image3.png" style="width: 554.00px; height: 312.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c8"><span>This leads to the second idea of the proposed architecture: judiciously applying dimension reductions and projections wherever the computational requirements would increase too much otherwise. That is, 1&times;1 convolutions are used to compute reductions before the expensive 3&times;3 and 5&times;5 convolutions. Besides being used as reductions, they also include the use of rectified linear activation which makes them dual-purpose.</span><span class="c1">&nbsp;</span></p>
<h2 class="c2" id="h.dojuwugy0uq0"><span class="c29">Use of 1&times;1 convolutions</span></h2>
<p class="c8"><span>Suppose we need to perform 5&times;5 convolution </span><span class="c40"><b>without the use of 1&times;1 convolution</b></span><span>&nbsp;</span><span class="c1">as</span></p>
<p class="c42"><span class="c1">Below</span></p>
<p class="c43"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 422.00px; height: 163.00px;"><img alt="" src="images/image2.png" style="width: 422.00px; height: 163.00px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c42"><span>Number of operations = (14 &times; 14 &times; 48) &times; (5 &times; 5 &times; 480) = 112.9M</span></p>
<p class="c42"><span class="c11"><b>With the use of 1&times;1 convolution:</b></span></p>
<p class="c42"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 149.33px;"><img alt="" src="images/image10.png" style="width: 624.00px; height: 149.33px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c8"><span class="c1">Number of operations for 1&times;1 = (14 &times; 14 &times; 16) &times; (1 &times; 1 &times; 480) = 1.5M</span></p>
<p class="c8"><span class="c1">Number of operations for 5&times;5 = (14 &times; 14 &times; 48) &times; (5 &times; 5 &times; 16) = 3.8M</span></p>
<p class="c8"><span class="c1">Total number of operations = 1.5M + 3.8M = 5.3M</span></p>
<p class="c8"><span class="c1">which is much much smaller than 112.9M !</span></p>
<h2 class="c2" id="h.5duee7gs1q9p"><span class="c29">Global Average Pooling</span></h2>
<p class="c8"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 242.67px;"><img alt="" src="images/image5.png" style="width: 624.00px; height: 242.67px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c8"><span>Previously, </span><span class="c40"><b>fully connected (FC) layers</b></span><span class="c1">&nbsp;were used at the end of a network, such as in AlexNet. All inputs are connected to each output.</span></p>
<p class="c8"><span class="c11"><b>Number of weights (connections) above = 7&times;7&times;1024&times;1024 = 51.3M</b></span></p>
<p class="c8"><span>In GoogLeNet, </span><span class="c40"><b>global average pooling</b></span><span class="c1">&nbsp;is used nearly at the end of the network by averaging each feature map from 7&times;7 to 1&times;1, as in the figure above.</span></p>
<p class="c8"><span class="c11"><b>Number of weights = 0</b></span></p>
<p class="c8"><span class="c1">And authors found that a move from FC layers to average pooling improved the top-1 accuracy by about 0.6%.</span></p>
<p class="c8"><span>This is the idea from </span><span class="c40"><b>NIN</b></span><span class="c1">&nbsp;which can be less prone to overfitting.</span></p>
<p class="c8 c26"><span class="c1"></span></p>
<p class="c8 c26"><span class="c1"></span></p>
<p class="c8 c26"><span class="c1"></span></p>
<p class="c8"><span>We may think that, when dimension is reduced, we are actually working on the mapping from high dimension to low dimension in a non-linearity way. In contrast, for PCA, it performs linear dimension reduction.</span></p>
<p class="c8"><span class="c1">In general, an Inception network is a network consisting of modules of the above type stacked upon each other, with occasional max-pooling layers with stride 2 to halve the resolution of the grid. it seemed beneficial to start using Inception modules only at higher layers while keeping the lower layers in traditional convolutional fashion.</span></p>
<h1 class="c19" id="h.ryjvyjif0y1u"><span class="c32">GoogLeNet</span></h1>
<p class="c8"><span class="c1">This name is an homage to Yann LeCuns pioneering LeNet 5 network</span></p>
<p class="c8"><span style="overflow: hidden; display: inline-block; margin: 0.00px 0.00px; border: 0.00px solid #000000; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px); width: 624.00px; height: 329.33px;"><img alt="" src="images/image8.png" style="width: 624.00px; height: 329.33px; margin-left: 0.00px; margin-top: 0.00px; transform: rotate(0.00rad) translateZ(0px); -webkit-transform: rotate(0.00rad) translateZ(0px);" title=""></span></p>
<p class="c8"><span class="c1">All the convolutions, including those inside the Inception modules, use rectified linear activation. The size of the receptive field in our network is 224&times;224 taking RGB color channels with mean subtraction. &ldquo;#3&times;3 reduce&rdquo; and &ldquo;#5&times;5 reduce&rdquo; stands for the number of 1&times;1 filters in the reduction layer used before the 3&times;3 and 5&times;5 convolutions.</span></p>
<p class="c8"><span class="c1">The network is 22 layers deep when counting only layers with parameters (or 27 layers if we also count pooling). The overall number of layers (independent building blocks) used for the construction of the network is about 100. A move from fully connected layers to average pooling improved the top-1 accuracy by about 0.6%, however the use of dropout remained essential even after removing the fully connected layers.</span></p>
<h1 class="c19" id="h.3vyyjmdbblgl"><span class="c32">Training Methodology</span></h1>
<p class="c8"><span class="c1">Authors used CPU based implementation only, a rough estimate suggests that the GoogLeNet network could be trained to converge using few high-end GPUs within a week, the main limitation being the memory usage. Training used asynchronous stochastic gradient descent with 0.9 momentum, fixed learning rate schedule (decreasing the learning rate by 4% every 8 epochs). Sampling of various sized patches of the image whose size is distributed evenly between 8% and 100% of the image area and whose aspect ratio is chosen randomly between 3/4 and 4/3. Also, the photometric distortions by Andrew Howard were useful to combat overfitting to some extent. In addition, they use random interpolation methods for resizing relatively late and in conjunction with other hyperparameter changes, so we could not tell definitely whether the final results were affected positively by their use.</span></p>
<p class="c8"><span class="c1">As we can see there are some intermediate softmax branches at the middle, they are used for training only. These branches are auxiliary classifiers which consist of:</span></p>
<b>
<p class="c8"><span class="c11">5&times;5 Average Pooling (Stride 3)</span></p>
<p class="c8"><span class="c11">1&times;1 Conv (128 filters)</span></p>
<p class="c8"><span class="c11">1024 FC</span></p>
<p class="c8"><span class="c11">1000 FC</span></p>
<p class="c8"><span class="c11">Softmax</span></p>
</b>
<p class="c8"><span class="c1">The loss is added to the total loss, with weight 0.3.</span></p>
<b><p class="c8"><span class="c11">Authors claim it can be used for combating gradient vanishing problems, also providing regularization.</span></p></b>
<p class="c8"><span class="c1">And it is NOT used in testing or inference time.</span></p>
<p class="c8 c26"><span class="c1"></span></p>
<h1 class="c19" id="h.5sn121d17mhl"><span class="c32">ILSVRC 2014 Classification Challenge Setup and Results</span></h1>
<p class="c8"><span class="c1">Authors independently trained 7 versions of the same GoogLeNet model (including one wider version), and performed ensemble prediction with them. These models were trained with the same initialization (even with the same initial weights, mainly because of an oversight) and learning rate policies, and they only differ in sampling methodologies and the random order in which they see input images.</span></p>
<p class="c8"><span class="c1">During testing, they resize the image to 4 scales where the shorter dimension is 256, 288, 320 and 352 respectively, take the left, center and right square of these resized images. For each square, they then take the 4 corners and the center 224&times;224 crop as well as the square resized to 224&times;224, and their mirrored versions. This results in 4&times;3&times;6&times;2 = 144 crops per image. </span></p>
<p class="c8"><span class="c1">The softmax probabilities are averaged over multiple crops and over all the individual classifiers to obtain the final prediction. In experiments we analyzed alternative approaches on the validation data, such as max pooling over crops and averaging over classifiers, but they lead to inferior performance than the simple averaging. </span></p>
<p class="c8"><span class="c1">Their final submission in the challenge obtains a top-5 error of 6.67% on both the validation and testing data, ranking the first among other participants. This is a 56.5% reduction compared to the SuperVision approach , and about 40% relative reduction compared to the Clarifai, both of which used external data for training the classifiers.</span></p>
<h2 class="c2" id="h.y3tdg19vwy4i"><span class="c29">Results</span></h2>
<a id="t.6d337cedc680934f615e81031c2c31b8df8b86a3"></a><a id="t.0"></a>
<table class="c37">
   <tbody>
      <tr class="c0">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Team</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Year</span></p>
         </td>
         <td class="c13" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Place</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Error (top-5)</span></p>
         </td>
         <td class="c5" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Uses external data</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">SuperVision </span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2012</span></p>
         </td>
         <td class="c13" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1st</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">16.4%</span></p>
         </td>
         <td class="c5" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">SuperVision</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2012</span></p>
         </td>
         <td class="c13" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1st</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">15.3%</span></p>
         </td>
         <td class="c5" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">Imagenet 22k</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">Clarifai</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2013</span></p>
         </td>
         <td class="c13" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1st</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">11.7%</span></p>
         </td>
         <td class="c5" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">Clarifai</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2013</span></p>
         </td>
         <td class="c13" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1st</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">11.2%</span></p>
         </td>
         <td class="c5" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">Imagenet 22k</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">MSRA</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2014</span></p>
         </td>
         <td class="c13" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">3rd</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">7.35%</span></p>
         </td>
         <td class="c5" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">VGG</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2014</span></p>
         </td>
         <td class="c13" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2nd</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">7.32%</span></p>
         </td>
         <td class="c5" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c22" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">GoogLeNet</span></p>
         </td>
         <td class="c4" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2014</span></p>
         </td>
         <td class="c13" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1st</span></p>
         </td>
         <td class="c18" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">6.67%</span></p>
         </td>
         <td class="c5" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c20"><span class="c1"></span></p>
<a id="t.f92555d49fa60fa7d7ddc7c58d65d112222384ba"></a><a id="t.1"></a>
<table class="c37">
   <tbody>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Number of models</span></p>
         </td>
         <td class="c31" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Number of Crops</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Cost</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Top-5 error</span></p>
         </td>
         <td class="c35" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">compared to base</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1</span></p>
         </td>
         <td class="c31" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">10.07%</span></p>
         </td>
         <td class="c35" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">base</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1</span></p>
         </td>
         <td class="c31" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">10</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">10</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">9.15%</span></p>
         </td>
         <td class="c35" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">-0.92%</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1</span></p>
         </td>
         <td class="c31" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">144</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">144</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">7.89%</span></p>
         </td>
         <td class="c35" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">-2.18%</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">7</span></p>
         </td>
         <td class="c31" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">7</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">8.09%</span></p>
         </td>
         <td class="c35" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">-1.98%</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">7</span></p>
         </td>
         <td class="c31" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">10</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">70</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">7.62%</span></p>
         </td>
         <td class="c35" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">-2.45%</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">7</span></p>
         </td>
         <td class="c31" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">144</span></p>
         </td>
         <td class="c21" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1008</span></p>
         </td>
         <td class="c33" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">6.67%</span></p>
         </td>
         <td class="c35" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">-3.45%</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c8 c26"><span class="c1"></span></p>
<h1 class="c19" id="h.wwl4wtjinvta"><span class="c32">ILSVRC 2014 Detection Challenge Setup and Results</span></h1>
<p class="c8"><span class="c1">The approach taken by GoogLeNet for detection is similar to the R-CNN, but is augmented with the Inception model as the region classifier. Additionally, the region proposal step is improved by combining the Selective Search approach with multi-box predictions for higher object bounding box recall. In order to cut down the number of false positives, the superpixel size was increased by 2&times;. This halves the proposals coming from the selective search algorithm. They added back 200 region proposals coming from multi-box resulting, in total, in about 60% of the proposals used by, while increasing the coverage from 92% to 93%. The overall effect of cutting the number of proposals with increased coverage is a 1% improvement of the mean average precision for the single model case. Finally, they use an ensemble of 6 ConvNets when classifying each region which improves results from 40% to 43.9% accuracy. </span></p>
<h2 class="c2" id="h.s0kl2rsrxcyi"><span class="c29">Results</span></h2>
<p class="c8 c26"><span class="c1"></span></p>
<a id="t.e17f678be7e5c41650b8820a3e15ac2c8edeab6a"></a><a id="t.2"></a>
<table class="c37">
   <tbody>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Team</span></p>
         </td>
         <td class="c39" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Year</span></p>
         </td>
         <td class="c15" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Place</span></p>
         </td>
         <td class="c6" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">mAP</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">External data</span></p>
         </td>
         <td class="c16" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">ensemble</span></p>
         </td>
         <td class="c41" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">approach</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">UvA-Euvision</span></p>
         </td>
         <td class="c39" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2013</span></p>
         </td>
         <td class="c15" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1st</span></p>
         </td>
         <td class="c6" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">22.6%</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">none</span></p>
         </td>
         <td class="c16" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">?</span></p>
         </td>
         <td class="c41" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">Fisher vectors</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">Deep Insight</span></p>
         </td>
         <td class="c39" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2014</span></p>
         </td>
         <td class="c15" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">3rd</span></p>
         </td>
         <td class="c6" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">40.5%</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">ImageNet 1k</span></p>
         </td>
         <td class="c16" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">3</span></p>
         </td>
         <td class="c41" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">CNN</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">CUHK DeepID-Net</span></p>
         </td>
         <td class="c39" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2014</span></p>
         </td>
         <td class="c15" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2nd</span></p>
         </td>
         <td class="c6" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">40.7%</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">ImageNet 1k</span></p>
         </td>
         <td class="c16" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">?</span></p>
         </td>
         <td class="c41" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">CNN</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c28" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">GoogLeNet</span></p>
         </td>
         <td class="c39" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">2014</span></p>
         </td>
         <td class="c15" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">1st</span></p>
         </td>
         <td class="c6" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">4.9%</span></p>
         </td>
         <td class="c24" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">ImageNet 1k</span></p>
         </td>
         <td class="c16" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">6</span></p>
         </td>
         <td class="c41" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">CNN</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c20"><span class="c1"></span></p>
<a id="t.afc05d1f8aff637329bb1addee2d221ae42c38e2"></a><a id="t.3"></a>
<table class="c37">
   <tbody>
      <tr class="c0">
         <td class="c7" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Team </span></p>
         </td>
         <td class="c10" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">mAP</span></p>
         </td>
         <td class="c23" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Contextual model</span></p>
         </td>
         <td class="c14" colspan="1" rowspan="1">
            <p class="c3"><span class="c11">Bounding box regression</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c7" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">Trimps-Soushen</span></p>
         </td>
         <td class="c10" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">31.6%</span></p>
         </td>
         <td class="c23" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
         <td class="c14" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">?</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c7" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">Berkeley Vision</span></p>
         </td>
         <td class="c10" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">34.5%</span></p>
         </td>
         <td class="c23" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
         <td class="c14" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">yes</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c7" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">UvA-Euvision </span></p>
         </td>
         <td class="c10" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">35.4%</span></p>
         </td>
         <td class="c23" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">?</span></p>
         </td>
         <td class="c14" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">?</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c7" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">CUHK DeepID-Net2</span></p>
         </td>
         <td class="c10" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">37.7%</span></p>
         </td>
         <td class="c23" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
         <td class="c14" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">?</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c7" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">GoogLeNet</span></p>
         </td>
         <td class="c10" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">38.2%</span></p>
         </td>
         <td class="c23" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
         <td class="c14" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">no</span></p>
         </td>
      </tr>
      <tr class="c0">
         <td class="c7" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">Deep Insight</span></p>
         </td>
         <td class="c10" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">40.2%</span></p>
         </td>
         <td class="c23" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">yes</span></p>
         </td>
         <td class="c14" colspan="1" rowspan="1">
            <p class="c3"><span class="c1">yes</span></p>
         </td>
      </tr>
   </tbody>
</table>
<p class="c8 c26"><span class="c1"></span></p>
</body></html>

For more detail report of GoogLeNet visit [here](https://github.com/DhruvMakwana/Computer_Vision/blob/master/GoogLeNet/GoogLeNet%20Detail%20Report.pdf)

To implement GoogLeNet in keras using python script run `GoogLeNet_Keras.py` or to implement AlexNet in keras using notebook run `GoogLeNet_Keras.ipynb`

To implement GoogLeNet in pytorch using python script run `GoogLeNet_pytorch.py` or to implement GoogLeNet in pytorch using notebook run `GoogLeNet_pytorch.ipynb`

<b>Note:</b> We have used Batch Normalization in keras version instead of LRN.
