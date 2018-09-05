using System;

namespace TrainingNetwork
{
    internal class NeuralNetwork
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

            this.ihPreviousBiases = new double[this.numOutput];
            this.ihPreviousWeightsDelta = this.MakeMatrix(this.numInput, this.numOutput);

            this.outputs = new double[this.numOutput];
            this.ohWeights = this.MakeMatrix(this.numHidden, this.numOutput);
            this.oBias = new double[this.numOutput];
            this.ohPreviousWeightsDelta = this.MakeMatrix(this.numHidden, this.numOutput);
            this.ohPreviousBiases = new double[this.numOutput];

            this.InitializeWeights();
        }

        private void InitializeWeights()
        {
            
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
            for(int i =0; i < result.Length; ++i)
            {
                result[i] = new double[numHidden];
            }

            return result;
        }

        internal void Train(double[][] trainData, int maxEpochs, double learningRate, double momentum)
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
            throw new NotImplementedException();
        }

        /// <summary>
        /// ComputeOutputs
        /// </summary>
        /// <param name="xValus"></param>
        private double[] ComputeOutputs(double[] xValus)
        {
            throw new NotImplementedException();
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
        internal double MeanSquaredError(double[][] trainingData)
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

        internal double Accuracy(double[][] testData)
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

        private int MaxIndex(double[] yValues)
        {
            int bigIndex = 0;
            double biggestVal = yValues[0];
            for (int i = 0; i < yValues.Length; ++i)
            {
                if (yValues[i] > biggestVal)
                {
                    biggestVal = yValues[i];
                }
            }

            return bigIndex;


        }


        private double MeanCrossEntropyError(double[][] trainData)
        {
            double sumErro = 0.0;
            double[] xValues = new double[numInput];
            double[] tValues = new double[numOutput]; // Last numOutput values;

            for(int i = 0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput); // get target
                double[] yValues = this.ComputeOutputs(xValues); // Compute output using current weights

                for(int j =0; j < numOutput; ++j)
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
           
        }

        internal double[] GetWeights()
        {
            throw new NotImplementedException();
        }
    }
}