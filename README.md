# tensorflow-LED-Identification
This code is used to identificate the LED's states.
This code copy from http://www.tensorfly.cn/.
It was used to identificate MNIST.
The aim of the code is finding the way to use my own data sets to identificate train sets.I want to use my own data sets.
Each input must be a one-dimensional vector.
You can change the structure of the CNN to improve the speed.but it will cut down the accuracy.So you should increase the epoches.It is common.I write this nonsense words only for learning English.hahahaha 
If you want to output the test set's judgment result.You could add "result = tf.argmax(y_conv, 1)" after the model of CNN,and add "res = sess.run(result, feed_dict={x: "(your test sets)", keep_prob: 1.0})" in the end of the every epoch or the end of the code.
The "y_conv" is the forward propagation's result.
