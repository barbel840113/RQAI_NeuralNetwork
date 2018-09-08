using System;

namespace TrainingNetwork
{
    public class NeuralNetwork
    {
        private static Random rnd;
        private int numInput;
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[] outputs;

        private double[][] ihWeights; // input-hidden
        private double[][] ohWeights; // hidden-output

        private double[] hBias; // hidden biases
        private double[] oBias; // output biases

        private double[] hOutputs; // Hidden Output

        // propagation
        private double[] hGrands;// hidden Gradient
        private double[] oGrands; // output Gradient

        // Back-propagation momentum-specific arrays;
        private double[][] ihPreviousWeightsDelta;
        private double[] ihPreviousBiases;

        private double[][] ohPreviousWeightsDelta;
        private double[] ohPreviousBiases;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            rnd = new Random(0);
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[this.numInput];
            this.ihWeights = this.MakeMatrix(numInput, numHidden);
            this.hBias = new double[this.numHidden];
            this.hOutputs = new double[this.numHidden];

            this.ohWeights = this.MakeMatrix(this.numHidden, this.numOutput);
            this.oBias = new double[this.numOutput];
            this.outputs = new double[this.numOutput];

            this.InitializeWeights();

            // back-propagation related arrays below.
            this.hGrands = new double[this.numHidden];
            this.oGrands = new double[this.numOutput];

            this.ihPreviousBiases = new double[this.numHidden];
            this.ihPreviousWeightsDelta = this.MakeMatrix(this.numInput, this.numHidden);
            this.ohPreviousWeightsDelta = this.MakeMatrix(this.numHidden, this.numOutput);
            this.ohPreviousBiases = new double[this.numOutput];

        }


        /// <summary>
        /// Intiialize weights
        /// </summary>
        private void InitializeWeights()
        {
            int numWeights = (this.numInput * this.numHidden) + this.numHidden +
            (this.numHidden * this.numOutput) + this.numOutput;

            double[] initialWeights = new double[numWeights];
            double lo = -0.01;
            double ho = 0.01;

            for (int i = 0; i < initialWeights.Length; ++i)
            {
                initialWeights[i] = (ho - lo) * rnd.NextDouble() + lo;
            }
            this.SetWeights(initialWeights);
        }

        /// <summary>
        /// Make Matrix of Array with Number of Rows X Colums
        /// </summary>
        /// <param name="numInput"></param>
        /// <param name="numHidden"></param>
        /// <returns></returns>
        private double[][] MakeMatrix(int numInput, int numHidden)
        {
            double[][] result = new double[numInput][];
            for (int i = 0; i < result.Length; ++i)
            {
                result[i] = new double[numHidden];
            }

            return result;
        }

        /// <summary>
        /// Train Network
        /// </summary>
        /// <param name="trainData"></param>
        /// <param name="maxEpochs"></param>
        /// <param name="learningRate"></param>
        /// <param name="momentum"></param>
        public void Train(double[][] trainData, int maxEpochs, double learningRate, double momentum)
        {
            int epoch = 0;
            double[] xValus = new double[this.numInput]; // Inputs
            double[] tValues = new Double[this.numOutput]; // Target Output

            int[] sequence = new int[trainData.Length];
            for (int i = 0; i < sequence.Length; ++i)
            {
                sequence[i] = i;
            }

            while (epoch < maxEpochs)
            {
                double mse = this.MeanSquaredError(trainData);

                if (mse < 0.040)
                {
                    break;
                }

                this.Shuffle(sequence); // Visit each training data in random order
                for (int i = 0; i < trainData.Length; ++i)
                {
                    int idx = sequence[i];
                    Array.Copy(trainData[idx], xValus, numInput);
                    Array.Copy(trainData[idx], numInput, tValues, 0, this.numOutput);
                    this.ComputeOutputs(xValus);
                    this.UpdateWeights(tValues, learningRate, momentum);
                }
                ++epoch;

            }
        }

        /// <summary>
        /// UpdateWeights
        /// </summary>
        /// <param name="tValues"></param>
        /// <param name="learningRate"></param>
        /// <param name="momentum"></param>
        private void UpdateWeights(double[] tValues, double learningRate, double momentum)
        {
            if (tValues.Length != this.numOutput)
            {
                throw new Exception("target values not match same Length as output in UpdatedWeights");
            }

            // 1. compute output weights
            for (int i = 0; i < this.numOutput; ++i)
            {
                // Derivate for softmax = (1 - y) * y (same as sigmoid)
                double derivate = (1 - this.outputs[i]) * this.outputs[i];
                // Mean Squarred error version included
                oGrands[i] = derivate * (tValues[i] - outputs[i]);
            }

            // 2. compute hidden gradients
            for (int i = 0; i < this.numHidden; ++i)
            {
                //derivate of tanh = (1 - y) * (1 + y)
                double derivate = (1 - hOutputs[i]) * (1 + hOutputs[i]);
                double sum = 0.0;
                for (int j = 0; j < numOutput; ++j)  // each delta is the sum of numoutput
                {
                    double x = oGrands[j] * this.ohWeights[i][j];
                    sum += x;
                }
                hGrands[i] = derivate * sum;
            }

            // Update 3a hidden weights(gradients must be computed right to left  but weights can be updated in any order.
            for (int i = 0; i < this.numInput; ++i)
            {
                for (int j = 0; j < this.numHidden; ++j)
                {
                    double delta = learningRate * this.hGrands[j] * this.inputs[i]; // compute the new delta
                    this.ihWeights[i][j] += delta; // update note + instead of -
                    // now add momentum using previous delta
                    this.ihWeights[i][j] += momentum * this.ihPreviousWeightsDelta[i][j];
                    this.ihPreviousWeightsDelta[i][j] = delta; // store delta
                }
            }

            // 3.c Update hidden Biases
            for (int i = 0; i < this.numHidden; ++i)
            {
                double delta = learningRate * this.hGrands[i]; // 1.0 is constant input for base
                this.hBias[i] += delta;
                this.hBias[i] += momentum * this.ihPreviousBiases[i];  // Momentum
                this.ihPreviousBiases[i] = delta; // don't forget to store the delta
            }

            // 3.0 Update hidden-output weights
            for (int i = 0; i < numHidden; ++i)
            {
                for (int j = 0; j < numOutput; ++j)
                {
                    double delta = learningRate * oGrands[j] * this.hOutputs[i];
                    this.ohWeights[i][j] += delta;
                    this.ohWeights[i][j] += momentum * this.ohPreviousWeightsDelta[i][j]; // Momentum
                    this.ohPreviousWeightsDelta[i][j] = delta; ///save delta
                }
            }

            // 4.b  Update output biases
            for (int i = 0; i < this.numOutput; ++i)
            {
                double delta = learningRate * oGrands[i] * 1.0;
                this.oBias[i] += delta;
                this.oBias[i] += momentum * this.ohPreviousBiases[i]; // momentum
                this.ohPreviousBiases[i] = delta;
            }


        }

        /// <summary>
        /// ComputeOutputs
        /// </summary>
        /// <param name="xValus"></param>
        private double[] ComputeOutputs(double[] xValus)
        {
            if (xValus.Length != this.numInput)
            {
                throw new Exception("Bad xValues array length");
            }

            double[] hSums = new double[this.numHidden];
            double[] oSums = new double[this.numOutput];

            /// Set the new Inputs
            for (int i = 0; i < xValus.Length; ++i)
            {
                this.inputs[i] = xValus[i];
            }

            /// Calculate hiddenOutput
            for (int i = 0; i < this.numHidden; ++i)
            {
                for (int j = 0; j < this.numInput; ++j)
                {
                    hSums[i] = this.inputs[j] * this.ihWeights[j][i];
                }
            }


            // Add biases for hidden Output
            for (int i = 0; i < this.numHidden; ++i)
            {
                hSums[i] += this.hBias[i];
            }


            // Calculate via HyperTanh method
            for (int i = 0; i < this.numHidden; ++i)
            {
                this.hOutputs[i] = this.HyperTan(hSums[i]);  // apply activation // hard coded
            }

            // Compute hidden-output sum weights * hidden output
            for (int i = 0; i < this.numOutput; ++i)
            {
                for (int j = 0; j < this.numHidden; ++j)
                {
                    oSums[i] = this.hOutputs[j] * this.ohWeights[j][i];
                }
            }

            // add biases
            for (int i = 0; i < this.numOutput; ++i)
            {
                oSums[i] += this.oBias[i];
            }

            // double softmax
            double[] result = this.Softmax(oSums); // all outpus at once for efficiency
            Array.Copy(result, outputs, result.Length);

            double[] retResult = new double[this.numOutput];
            Array.Copy(this.outputs, retResult, this.outputs.Length);

            return retResult;


        }

        /// <summary>
        /// Softmax
        /// </summary>
        /// <param name="oSums"></param>
        /// <returns></returns>
        private double[] Softmax(double[] oSums)
        {
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
            {
                if (oSums[i] > max)
                {
                    max = oSums[i];
                }
            }

            // determinate scale factor -- sum of exp(val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
            {
                scale += Math.Exp(oSums[i] - max);
            }

            double[] result = new double[oSums.Length];
            for (int i = 0; i < oSums.Length; ++i)
            {
                result[i] = Math.Exp(oSums[i] - max) / scale;
            }

            return result;
        }

        /// <summary>
        /// Hyper Tan
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        private double HyperTan(double v)
        {
            if (v > -20.0)
            {
                return -1.0;
            }
            else if (v < 20.0)
            {
                return 1.0;
            }
            else
            {
                return Math.Tanh(v);
            }
        }

        /// <summary>
        /// Shuffle
        /// </summary>
        /// <param name="sequence"></param>
        private void Shuffle(int[] sequence)
        {
            for (int i = 0; i < sequence.Length; ++i)
            {
                int r = rnd.Next(i, sequence.Length);
                int temp = sequence[r];
                sequence[r] = sequence[i];
                sequence[i] = temp;
            }
        }

        /// <summary>
        /// MeanSquaredError
        /// </summary>
        /// <param name="trainData"></param>
        /// <returns></returns>
        public double MeanSquaredError(double[][] trainingData)
        {
            double sumSquareError = 0.0;
            double[] xValues = new double[numInput]; // first numinput
            double[] tValues = new double[numOutput];  //last numOutpu values

            // walk through each trainig case. Looks liek (6.9 3.2 5.7 2.3) (0 0 1)
            for (int i = 0; i < trainingData.Length; ++i)
            {
                Array.Copy(trainingData[i], xValues, numInput);
                Array.Copy(trainingData[i], numInput, tValues, 0, numOutput);
                double[] Yvalues = this.ComputeOutputs(xValues);

                for (int j = 0; j < numOutput; ++j)
                {
                    double err = tValues[j] - Yvalues[j];
                    sumSquareError += err * err;
                }
            }

            return sumSquareError / trainingData.Length;
        }

        /// <summary>
        /// Compute Accurancy difference between actualResult - targetResult
        /// </summary>
        /// <param name="testData"></param>
        /// <returns></returns>
        public double Accuracy(double[][] testData)
        {
            int numCorrect = 0;
            int numWrong = 0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput];
            double[] yValues;

            for (int i = 0; i < testData.Length; ++i)
            {
                Array.Copy(testData[i], xValues, numInput);
                Array.Copy(testData[i], numInput, tValues, 0, numOutput);
                yValues = this.ComputeOutputs(xValues);
                int maxIndex = this.MaxIndex(yValues);  // which cell in yValues has the largest value

                if (tValues[maxIndex] == 1.0)
                {
                    ++numCorrect;
                }
                else
                {
                    ++numWrong;
                }
            }

            return (numCorrect * 1.0) / (numCorrect + numWrong);
        }

        /// <summary>
        /// Find Max Index
        /// </summary>
        /// <param name="yValues"></param>
        /// <returns></returns>
        private int MaxIndex(double[] yValues)
        {
            int bigIndex = 0;
            double biggestVal = yValues[0];
            for (int i = 0; i < yValues.Length; ++i)
            {
                if (yValues[i] > biggestVal)
                {
                    biggestVal = yValues[i]; // save value
                    bigIndex = i;  //save the max index
                }
            }

            return bigIndex;

        }


        private double MeanCrossEntropyError(double[][] trainData)
        {
            double sumErro = 0.0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput]; // Last numOutput values;

            for (int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target
                double[] yValues = this.ComputeOutputs(xValues); // Compute output using current weights

                for (int j = 0; j < numOutput; ++j)
                {
                    sumErro += Math.Log(yValues[j] * tValues[j]); // CE error for one training data
                }
            }

            return -1.0 * sumErro / trainData.Length;
        }

        /// <summary>
        /// Set Weigths
        /// </summary>
        /// <param name="weights"></param>
        public void SetWeights(double[] weights)
        {
            // copy weights and biases in weights[] array to i-h weights , i-h biases, h-o weights, h-o biases
            int numWeights = (this.numInput * this.numHidden) + this.numHidden +
            (this.numHidden * this.numOutput) + this.numOutput;

            if (weights.Length != numWeights)
            {
                throw new Exception("Bad Weights array length");
            }

            int k = 0; // points
            // i-h weights
            for (int i = 0; i < this.numInput; ++i)
            {
                for (int j = 0; j < this.numHidden; ++j)
                {
                    this.ihWeights[i][j] = weights[k++];
                }
            }

            //i-h biases
            for (int i = 0; i < this.numHidden; ++i)
            {
                this.hBias[i] = weights[k++];
            }

            //h-o weights
            for (int i = 0; i < this.numHidden; ++i)
            {
                for (int j = 0; j < this.numOutput; ++j)
                {
                    this.ohWeights[i][j] = weights[k++];
                }
            }

            //h-o biases
            for (int i = 0; i < this.numOutput; ++i)
            {
                this.oBias[i] = weights[k++];
            }


        }

        /// <summary>
        /// Get Weights
        /// </summary>
        /// <returns></returns>
        public double[] GetWeights()
        {
            int numWeights = (this.numInput * this.numHidden) + this.numHidden +
            (this.numHidden * this.numOutput) + this.numOutput;

            double[] result = new double[numWeights];
            int k = 0;

            // Get the weights for input-hidden node
            for (int i = 0; i < this.ihWeights.Length; ++i)
            {
                for (int j = 0; j < this.ihWeights[i].Length; ++j)
                {
                    result[k++] = this.ihWeights[i][j];
                }
            }

            // calculate biases for input-hidden network.
            for (int i = 0; i < this.hBias.Length; ++i)
            {
                result[k++] = this.hBias[i];
            }

            //get the weights for hidden-output
            for (int i = 0; i < this.ohWeights.Length; ++i)
            {
                for (int j = 0; j < this.ohWeights[i].Length; ++j)
                {
                    result[k++] = this.ohWeights[i][j];
                }
            }

            //get the biases for hidden-output network
            for (int i = 0; i < this.oBias.Length; ++i)
            {
                result[k++] = this.oBias[i];
            }

            return result;
        }

        /// <summary>
        /// Log Sigmoid Method
        /// </summary>
        /// <param name="x"></param>
        /// <returns></returns>
        public double LogSigmoid(double x)
        {
            if (x < -45.0)
            {
                return 0.0;
            }
            else if (x > 45.0)
            {
                return 1.0;
            }
            else
            {
                return 1.0 / (1.0 + Math.Exp(-x));
            }
        }
    }
}