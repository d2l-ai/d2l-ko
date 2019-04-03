# From Dense Layers to Convolutions

So far we learned the basics of designing Deep Networks. Indeed, for someone dealing only with generic data, the previous sections are probably sufficient to train and deploy such a network sufficiently. There is one caveat, though - just like most problems in statistics, networks with many parameters either require a lot of data or a lot of regularization. As a result we cannot hope to design sophisticated models in most cases.

지금까지 우리는 딥 네트워크 디자인의 기본적인 내용을 배웠습니다. 사실은 일반적인 데이터만을 다루는 경우라면, 지금까지 배운 내용으로 여러분은 충분히 네트워크를 학습시키고 배포할 수 있습니다. 하지만 통계의 대부부의 문제들처럼 많은 파라메터를 갖는 네트워크는 아주 많은 양의 데이터나 많은 정규화(regularization)가 필요하다는 점을 유의해야합니다. 이때문에, 대부분의 경우에 복잡한 모델을 설계하는 것이 쉽지 않습니다.

For instance, consider the seemingly task of distinguishing cats from dogs. We decide to use a good camera and take 1 megapixel photos to ensure that we can really distinguish both species accurately. This means that the *input* into a network has 1 million dimensions. Even an aggressive reduction to 1,000 dimensions after the first layer means that we need $10^9$ parameters. Unless we have ㅊ amounts of data (billions of images of cats and dogs), this is mission impossible. Add in subsequent layers and it is clear that this approach is infeasible.

겉모습으로 개와 고양이를 구분하는 일을 예로 들어보겠습니다. 두 종류를 정확하기 구분하기 위해서 좋은 카메라를 마련해서 1메가픽셀 해상도의 사진을 찍습니다. 이것이 의미하는 것은 네트워크에 대입할 *입력*이 100만 차원이라는 것입니다. 첫번째 래이어가  1000차원으로 결과를 아무 많이 줄이는 경우에도  $10^9$ 개의 파라메터가 필요합니다. 데이터 양이 아주 풍부하지 않다면 (수억장의 개와 고양이 사진들) 이 과제는 불가능합니다. 만약 래이어를 더 추가한다면, 완전히 실현 불가능한 일이 될 것입니다.

The avid reader might object to the rather absurd implications of this argument by stating that 1 megapixel resolution is not necessary. But even if we reduce it to a paltry 100,000 pixels, that's still $10^8$ parameters. A corresponding number of images of training data is beyond the reach of most statisticians. In fact, the number exceeds the population of dogs and cats in all but the largest countries! Yet both humans and computers are able to distinguish cats from dogs quite well, often after only a few hundred images. This seems to contradict our conclusions above. There is clearly something wrong in the way we are approaching the problem. Let's find out.

**어떤 독자들은 1메가 픽셀 해당도가 필요없다고 주장할 수도 있습니다.** 하지만 해상도를 100,000 픽셀로 줄여도 여전히 $10^8$ 개의 파라메터가 필요합니다. 이 네트워크를 학습시키기 위해서 필요한 이미지 개수는 대다수 통계학자가 얻을 수 있는 범위를 훨신 넘어섭니다. 사실, 가장 큰 나라에 있는 모든 개와 고양이를 합한 수보다 큽니다. 하지만, 수백장의 이미지만으로도 사람과 컴퓨터는 개와 고양이를 잘 구분할 수 있는데, 이는 앞에서 이야기한 내용과 모순되어 보입니다. 그렇다면 우리의 접근 방법에 문제가 있다는 것을 의마하는데, 지금부터 찾아보겠습니다.

## Invariances

Imagine that you want to detect an object in an image. It is only reasonable to assume that the location of the object shouldn't matter too much to determine whether the object is there. We should assume that we would recognize an object wherever it is in an image. This is true within reason - pigs usually don't fly and planes usually don't swim. Nonetheless, we would still recognize a flying pig, albeit possibly after a double-take. This fact manifests itself e.g. in the form of the children's game 'Where is Waldo'. In it, the goal is to find a boy with red and white striped clothes, a striped hat and black glasses within a panoply of activity in an image. Despite the rather characteristic outfit this tends to be quite difficult, due to the large amount of confounders. The image below, on the other hand, makes the problem particularly easy.

이미지에서 어떤 사물을 찾고자 한다고 생각해보세요. 사물이 있는지 여부를 결정하는데, 사물의 위치는 중요하기 않다고 가정해봅시다. 즉, 사물이 이미지의 어느 위치에 있건 그 사물을 인식하면된다고 가정하겠습니다. **This is true within reason - pigs usually don't fly and planes usually don't swim. Nonetheless, we would still recognize a flying pig, albeit possibly after a double-take.** ….

![](../img/waldo.jpg)

There are two key principles that we can deduce from this slightly frivolous reasoning:

다소 느슨한 생각을 해보면 다음과 두가지 주요 원칙을 추론해볼 수 있습니다.

1. Object detectors should work the same regardless of where in the image an object can be found. In other words, the 'waldoness' of a location in the image can be assessed (in first approximation) without regard of the position within the image. (Translation Invariance)
1. Object detection can be answered by considering only local information. In other words, the 'waldoness' of a location can be assessed (in first approximation) without regard of what else happens in the image at large distances. (Locality)
1. 객체 탐지기는 사물이 이미지의 어느 위치에 있는지 상관없이 동작해야합니다. **다르게 말하면,  왈도가 이미지내의 어디에 있든지 상관없이(Translation Invariance) 왈도의 위치를 알아낼 수 있어야 합니다 (in first approximation)**
1. 객체 탐지는 지협적인 정보만 고려해서 답을 줘야합니다. 즉, 이미지안에서 멀리 떨어져있는 다른 것들과는 상관없이 (Locality) 왈도의 위치를 알아낼 수 있어야 합니다.

Let's see how this translates into mathematics.

이것이 수학적으로 어떻게 해석되는지 보겠습니다.

## Constraining the MLP

In the following we will treat images and hidden layers as two-dimensional arrays. I.e. $x[i,j]$ and $h[i,j]$ denote the position $(i,j)​$ in an image. Consequently we switch from weight matrices to four-dimensional weight tensors. In this case a dense layer can be written as follows:

$$h[i,j] = \sum_{k,l} W[i,j,k,l] \cdot x[k,l] =
\sum_{a, b} V[i,j,a,b] \cdot x[i+a,j+b]$$

The switch from $W$ to $V$ is entirely cosmetic (for now) since there is a one to one correspondence between coefficients in both tensors. We simply re-index the subscripts $(k,l)$ such that $k = i+a$ and $l = j+b$. In other words we set $V[i,j,a,b] = W[i,j,i+a, j+b]$. The indices $a, b$ run over both positive and negative offsets, covering the entire image. For any given location $(i,j)$ in the hidden layer $h[i,j]$ we compute its value by summing over pixels in $x$, centered around $(i,j)$ and weighted by $V[i,j,a,b]$.

Now let's invoke the first principle we established above - *translation invariance*. This implies that a shift in the inputs $x$ should simply lead to a shift in the activations $h$. This is only possible if $V$ doesn't actually depend on $(i,j)$, that is, we have $V[i,j,a,b] = V[a,b]$. As a result we can simplify the definition for $h$.

$$h[i,j] = \sum_{a, b} V[a,b] \cdot x[i+a,j+b]$$

This is a convolution! We are effectively weighting pixels $(i+a, j+b)$ in the vicinity of $(i,j)$ with coefficients $V[a,b]$ to obtain the value $h[i,j]$. Note that $V[a,b]$ needs a lot fewer coefficients than $V[i,j,a,b]$. For a 1 megapixel image it has at most 1 million coefficients. This is 1 million fewer parameters since it no longer depends on the location within the image. We have made significant progress!

Now let's invoke the second principle - *locality*. In the problem of detecting Waldo we shouldn't have to look very far away from $(i,j)$ in order to glean relevant information to assess what is going on at $h[i,j]$. This means that outside some range $|a|, |b| > \Delta$ we should set $V[a,b] = 0$. Equivalently we can simply rewrite $h[i,j]$ as

$$h[i,j] = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} V[a,b] \cdot x[i+a,j+b]$$

This, in a nutshell is the convolutional layer. The difference to the fully connected network is dramatic. While previously we might have needed $10^8$ or more coefficients, we now only need $O(\Delta^2)$ terms. The price that we pay for this drastic simplification is that our network will be translation invariant and that we are only able to take local information into account.

## Convolutions

Let's briefly review why the above operation is called a convolution. In math the convolution between two functions, say $f, g: \mathbb{R}^d \to R$ is defined as

$$[f \circledast g](x) = \int_{\mathbb{R}^d} f(z) g(x-z) dz$$

That is, we measure the overlap beween $f$ and $g$ when both functions are shifted by $x$ and 'flipped'. Whenever we have discrete objects the integral turns into a sum. For instance, for vectors defined on $\ell_2$, i.e. the set of square summable infinite dimensional vectors with index running over $\mathbb{Z}$ we obtain the following definition.

$$[f \circledast g](i) = \sum_a f(a) g(i-a)$$

For two-dimensional arrays we have a corresponding sum with indices $(i,j)$ for $f$ and $(i-a, j-b)$ for $g$ respectively. This looks almost the same in the definition above, with one major difference. Rather than using $(i+a, j+b)$ we are using the difference instead. Note, though, that this distinction is mostly cosmetic since we can always match the notation by using $\tilde{V}[a,b] = V[-a, -b]$ to obtain $h = x \circledast \tilde{V}$. Note that the original definition is actually a *cross correlation*. We will come back to this in the following section.


## Waldo Revisited

Let's see what this looks like if we want to build an improved Waldo detector. The convolutional layer picks windows of a given size and weighs intensities according to the mask $V$. We expect that wherever the 'waldoness' is highest, we will also find a peak in the hidden layer activations.

![](../img/waldo-mask.jpg)

There's just a problem with this approach: so far we blissfully ignored that images consist of 3 channels - red, green and blue. In reality images are thus not two-dimensional objects but three-dimensional tensors, e.g. of $1024 \times 1024 \times 3$ pixels. We thus index $\mathbf{x}$ as $x[i,j,k]$. The convolutional mask has to adapt accordingly. Instead of $V[a,b]$ we now have $V[a,b,c]$.

The last flaw in our reasoning is that this approach generates only one set of activations. This might not be great if we want to detect Waldo in several steps. We might need edge detectors, detectors for different colors, etc.; In short, we want to retain some information about edges, color gradients, combinations of colors, and a great many other things. An easy way to address this is to allow for *output channels*. We can take care of this by adding a fourth coordinate to $V$ via $V[a,b,c,d]$. Putting all together we have:

$$h[i,j,k] = \sum_{a = -\Delta}^{\Delta} \sum_{b = -\Delta}^{\Delta} \sum_c V[a,b,c,k] \cdot x[i+a,j+b,c]$$

This is the definition of a convolutional neural network layer. There are still many operations that we need to address. For instance, we need to figure out how to combine all the activations to a single output (e.g. whether there's a Waldo in the image). We also need to decide how to compute things efficiently, how to combine multiple layers, and whether it is a good idea to have many narrow or a few wide layers. All of this will be addressed in the remainder of the chapter. For now we can bask in the glory having understood why convolutions exist in principle.

## Summary

* Translation invariance in images implies that all patches of an image will be treated in the same manner.
* Locality means that only a small neighborhood of pixels will be used for computation.
* Channels on input and output allows for meaningful feature analysis.

## Problems

1. Assume that the size of the convolution mask is $\Delta = 0$. Show that in this case the convolutional mask implements an MLP independently for each set of channels.
1. Why might translation invariance not be a good idea after all? Does it make sense for pigs to fly?
1. What happens at the boundary of an image?
1. Derive an analogous convolutional layer for audio.
1. What goes wrong when you apply the above reasoning to text? Hint - what is the structure of language?
1. Prove that $f \circledast g = g \circledast f$.

## Scan the QR Code to [Discuss](https://discuss.mxnet.io/t/2348)

![](../img/qr_why-conv.svg)
