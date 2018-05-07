import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# ==== CREATION OF A 2D GAUSSIAN KERNEL ====
x = tf.linspace(-3.0, 3.0, 100)

# The 1 dimensional gaussian takes two parameters, the mean value, and the standard deviation
mean = 0.0
sigma = 1.0
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))) * (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))

# Let's store the number of values in our Gaussian curve.
ksize = z.get_shape().as_list()[0]

# Let's multiply the two to get a 2d gaussian
z_2d = tf.matmul(tf.reshape(z, [ksize, 1]), tf.reshape(z, [1, ksize]))

# Execute the graph
sess = tf.Session()
plt.imshow(z_2d.eval(session=sess))
plt.show()

# ==== CONVOLVING AN IMAGE WITH A 2D GAUSSIAN ====
# Let's load a grayscale image
img = data.camera().astype(np.float32)
plt.imshow(img, cmap='gray')
print(img.shape)

# For image convolution in TF, we need our images to be 4-dimensional : Number x Height x Width x Channels
# We could use the numpy reshape function to reshape our numpy array
img_4d = img.reshape([1, img.shape[0], img.shape[1], 1])
print(img_4d.shape)

# but since we'll be using TensorFlow, we can use the TensorFlow reshape function:
img_4d = tf.reshape(img, [1, img.shape[0], img.shape[1], 1])
print(img_4d)
# Use get_shape() on a TF tensor
print(img_4d.get_shape())
print(img_4d.get_shape().as_list())

# Reshape the 2d Gaussian kernel to TensorFlow's required 4d format: H x W x I x O
# Kernel Height, Kernel width, number of input channels (=n_channels_image), number of output channels
# We'll keep the number of output channels equal to the number of input channels for now.
z_4d = tf.reshape(z_2d, [ksize, ksize, img_4d.get_shape().as_list()[3], img_4d.get_shape().as_list()[3]])
print(z_4d.get_shape().as_list())

# We can now use the kernel to convolve the image. Padding='SAME' to keep the image's original dimensions
convolved = tf.nn.conv2d(img_4d, z_4d, strides=[1, 1, 1, 1], padding='SAME')
res = convolved.eval(session=sess)
print(res.shape)

# MatPlotLib cannot handle plotting 4D images!  We'll have to convert this back to the original shape.
# There are a few ways we could do this.  We could plot by "squeezing" the singleton dimensions.
plt.imshow(np.squeeze(res), cmap='gray')
# Or we could specify the exact dimensions we want to visualize:
plt.imshow(res[0, :, :, 0], cmap='gray')
plt.show()

# ==== CREATING A GABOR KERNEL ====
# Let's modulate the Gaussian by a sine wave to obtain a Gabor kernel
# Calculate the sine of the x values
ys = tf.sin(x)
plt.figure()
plt.plot(ys.eval(session=sess))
plt.show()

# Convert the 1-dimensional vector a matrix : N x 1 (= transpose the vector)
ys = tf.reshape(ys, [ksize, 1])
# We now repeat this wave across a (ksize, ksize) matrix by using a multiplication of ones
ones = tf.ones((1, ksize))
wave = tf.matmul(ys, ones)
plt.imshow(wave.eval(session=sess), cmap='gray')
plt.show()
# We can now multiply the Gaussian kernel by this wave and get a Gabor kernel
gabor = tf.multiply(wave, z_2d)
plt.imshow(gabor.eval(session=sess), cmap='gray')
plt.show()

# ==== CONVOLVING AN IMAGE WITH THE GABOR KERNEL ====
# Let's now use placeholders to make our convolution reusable.
# Declare a placeholder which will become part of the TensorFlow graph, but
# which we have to later explicitly define whenever we run/evaluate the graph.
img = tf.placeholder(tf.float32, shape=[None, None], name='img')

# We'll reshape the 2d image to a 3-d tensor just like before:
# Except now we'll make use of another TensorFlow function, expand dims, which adds a
# singleton dimension at the axis we specify.
# We use it to reshape our H x W image to include a channel dimension of 1
# our new dimensions will end up being: H x W x 1
img_3d = tf.expand_dims(img, 2)
dims = img_3d.get_shape()
print(dims)

# And again to get: 1 x H x W x 1
img_4d = tf.expand_dims(img_3d, 0)
print(img_4d.get_shape().as_list())

# Let's create another set of placeholders for our Gabor's parameters:
mean = tf.placeholder(tf.float32, name='mean')
sigma = tf.placeholder(tf.float32, name='sigma')
ksize = tf.placeholder(tf.int32, name='ksize')

# Then finally redo the entire set of operations we've done to convolve our
# image, except with our placeholders
x = tf.linspace(-3.0, 3.0, ksize)
z = (tf.exp(tf.negative(tf.pow(x - mean, 2.0) / (2.0 * tf.pow(sigma, 2.0)))) * (1.0 / (sigma * tf.sqrt(2.0 * 3.1415))))
z_2d = tf.matmul(
  tf.reshape(z, tf.stack([ksize, 1])),
  tf.reshape(z, tf.stack([1, ksize]))
)
ys = tf.sin(x)
ys = tf.reshape(ys, tf.stack([ksize, 1]))
ones = tf.ones(tf.stack([1, ksize]))
wave = tf.matmul(ys, ones)
gabor = tf.multiply(wave, z_2d)
gabor_4d = tf.reshape(gabor, tf.stack([ksize, ksize, 1, 1]))

# And finally, convolve the two:
convolved = tf.nn.conv2d(img_4d, gabor_4d, strides=[1, 1, 1, 1], padding='SAME', name='convolved')
convolved_img = convolved[0, :, :, 0]

with tf.Session() as sess2:
    res = convolved_img.eval(feed_dict={img: data.camera(), mean: 0.0, sigma: 1.0, ksize: 100})
    plt.imshow(res, cmap='gray')
    plt.show()

    # Now instead of having to rewrite the entire graph, we can just specify different placeholders
    res = convolved_img.eval(feed_dict={
        img: data.camera(),
        mean: 0.0,
        sigma: 0.5,
        ksize: 32
      })
    plt.imshow(res, cmap='gray')
    plt.show()
