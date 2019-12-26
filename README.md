# Logistic-Regression-on-Cancer-Diagnosis
It's a classification technique and it's very simple and geometrically elegant algorithm. We can interpret this logistic regression using Geometrical, Probability and loss function. The main assumption of logistic regression is classes(target values) are almost or perfectly linearly seperable.

![img](https://media.geeksforgeeks.org/wp-content/uploads/estimated-regression-result.png)

If you consider the above image the red and green points are related to 2 classes and those outcomes are linearly seperable. As you aware the line of equation in 2D but when it comes higher dimensions we called the line as plane. We can write the plane eqaution as :
            (W.T)* x + b = 0
            
Where W is the weight vector and it contains weights of an each feature, x is an input vector and b is a intercept term. 
<h3>Geometric interpretation of logistic regression</h3>
Let, x(Predictor/Features/Independent variable), y (Response/Target/Dependent variable) be the dataset(D) i.e. D ∈ {xi , yi} of n data-points. Where xi ∈ ℝ^d for all ith observation, that is each xi’s is a real valued d dimension feature vector and yi ∈ (-1(-ve), 1(+ve)) that is each yi’s are either 1 or -1. The underlying assumption of logistic regression is data are almost (i.e. some +ve class points are in -ve class and vice-versa) or perfect (None of the points are mixed with other class) linearly separable (Figure -1)and our main objective is to find a line(in 2D) or plane/hyperplane(in 3D or more dimension) that can separate both the classes point as perfect as possible so that when it encounter with any new point It can easily classify, from which class point it belongs to. I.e. x and y are fixed because they are coming from training data so if we can find w (Normal) and b (bias/ y-intercept) then we can easily find a line or plane also called decision boundary. Here, we will focus on just two features (x1 and x2) only so that the intuition becomes easy. Although, in machine learning, it is almost impossible to have 2 or 3-dimension data.

![Figure 1](https://miro.medium.com/max/909/1*No261DjSDdX1-jXXVreW0g.png)
In Figure - 2, If we take any of the +ve class points and compute distance from a point to a plane (di = w^t*xi/||w||. let, norm (||w||) is 1). Since w and xi in the same side of the decision boundary then distance will be +ve. Now compute dj = w^t*xj since xj is opposite side of w then distance will be -ve. If we say, points which are in the same direction of w are all +ve points and the points which are in opposite direction of w are -ve points.

![Figure 2](https://miro.medium.com/max/512/1*cvba9slDh7DaLHHfdYOFzg.png)

Now, we could easily classify the -ve and +ve points using w^t*xi>0 then y =+1 and If w^t*xi < 0 then y = -1. While doing this we could do some mistake but it is okay because in real world we will never get data which are perfectly separable.
<h3>Observations:</h3>
Look at the figure 2 visually and observe all the listed points below-
* If yi = +1 means it is +ve data-points and w^t*xi > 0 i.e classifier(A mathematical function, implemented by a classification algorithm, that maps input data to a category.) is saying it is +ve points. So what happen, if yi*w^t*xi > 0 then it is correctly classified points because multiplying two +ve number will always be greater than 0.

* If yi = -1 means it is -ve data-points and w^t*xi < 0 i.e. classifier is saying it is -ve points. if yi * w^t*xi > 0 then it is correctly classified points because multiplying two -ve numbers will always be greater than zero. So, for both +ve and -ve points yi* w^t*xi > 0 this implies the model is correctly classifying the points xi.

* If yi = +1 and w^t*xi < 0 i.e. yi is +ve points but classifier is saying it is -ve then we will get -ve value. Which means actual class label is +ve but it is classified as -ve then this is miss-classified points.

* If yi = -1 and w^t*xi > 0. Which means actual class label is -ve but classified as +ve then it is miss-classified points( yi*w^t*xi < 0).

From above observations, we want our classifier to minimize miss-classification error. I.e. we want yi*w^t*xi to be greater than 0. Here, xi, yi are fixed because these are coming from data-set. As we change w, and b the sum will change and we want to find such w and b that maximize that sum given below.

![fig1](https://miro.medium.com/max/247/1*jvBRXl9wzF92kI6CDUkfmw.png)

<h3>Need for Logistic Function or “ S” shape curve or Sigmoid Function</h3>
Sigmoid function is a differentiable real function that is define for all real input and has non-negative derivative at each point. It is monotonic function which squashes value between 0 and 1. We will look at a very simple example where we will see how sum of signed distances (yi*w^t*xi) can be impacted by an erroneous/outlier points and we need to come up with another formulation which is less impacted by outlier.




Suppose in the left figure 3, the distance (d) from any point to decision boundary is 1 for all -ve side of decision boundary points and +ve side of decision boundary points, except an outlier point which is in the +ve side of the decision boundary and the distance is 100. If we compute the signed distance then it will be -90. In the right figure 3, the distance (d) from any point to decision boundary is 1 and their distances from each other is also 1. If we compute the signed distance then it will be 1. So, we have 5 miss-classified points (point is -ve but are in +ve side of the decision boundary) in right figure 3 and sum of signed distance is -90. In left figure 3, we have 1 miss-classified point and sum of signed distance is 1. And remember we wanted to maximize the sum of signed distances which is 1 in this case. So, If we choose sum of signed distance, in the presence of outlier, our prediction may not correct and we end up with worst model.

![fig2](https://miro.medium.com/max/853/1*D1cahB9JWj_OD-6d8AZT6g.png)
Figure 3
So, to avoid this problem we need another function that can be more robust than the maximizing signed distances . Such function we use here is called the sigmoid function and is define as

![fig3](https://miro.medium.com/max/192/1*RhUACLtPdXGMIw8_Q-iYwA.png)

So, we need to maximize the sigmoid function which is defined as

![fig4](https://miro.medium.com/max/333/1*SFJ4OcKaRhmX2nwOgyBN5w.png)

Maximizing some function f(x) is same as minimizing this function with -ve sign. I.e. argmax f(x) = argmin -f(x) and if we take log (we will discuss why use log in the loss minimization interpretation) then the final formulation becomes-

![fig5](https://miro.medium.com/max/435/1*cn36YTmxe1vmfeAxsefnZg.png)
