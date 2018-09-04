using System;

namespace TrainingNetwork
{
    internal class NeuralNetwork
    {
        private static Random rnd;
        private int numInput;
        private int numHidden;
        private int numOutput;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;
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
            for(int i =0; i < sequence.Length; ++i)
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
        internal double MeanSquaredError(double[][] trainData)
        {
            double sumSquareError = 0.0;
            double[] xValues = new double[numInput]; // first numinput
            double[] tValues = new double[numOutput];  //last numOutpu values

            // walk through each trainig case. Looks liek (6.9 3.2 5.7 2.3) (0 0 1)
            for(int i =0; i < trainData.Length; ++i)
            {
                Array.Copy(trainData[i], xValues, numInput);
                Array.Copy(trainData[i], numInput, tValues, 0, numOutput);
                double[] Yvalues = this.ComputeOutputs(xValues);
            }
        }
    }
}