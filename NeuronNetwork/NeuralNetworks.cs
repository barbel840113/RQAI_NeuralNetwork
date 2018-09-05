using System;
using System.Collections.Generic;
using System.Text;

namespace NeuronNetwork
{
    public class NeuralNetworks
    {
        public enum Activation { HyperTan, LogSigmoid, Softmax };
        private Activation hActivation;
        private Activation oActivation;

        private int numInput;
        private int numHidden;
        private int numOutput;

        private double[] inputs;
        private double[][] ihWeights;
        private double[] hBias;
        private double[] hOutpus;

        private double[][] ohWeights;
        private double[] oBiases;

        private double[] outputs;

        public NeuralNetworks(int numInput, int numHidden, int numOutput,
         Activation hActiv, Activation oActiv)
        {
            this.numInput = numInput;
            this.numHidden = numHidden;
            this.numOutput = numOutput;

            this.inputs = new double[numInput];
            this.ihWeights = MakeMatrix(numInput, numHidden);

            this.hBias = new double[numHidden];
            this.hOutpus = new double[numHidden];
            this.ohWeights = MakeMatrix(numHidden, numOutput);
            this.oBiases = new double[numOutput];
            this.outputs = new double[numOutput];

            this.hActivation = hActiv;
            this.oActivation = oActiv;
        }

        internal void SetWeights(double[] weights)
        {
            int numWeights = (this.numInput * this.numHidden) + this.numHidden
            + (this.numHidden * this.numOutput) + this.numOutput;

            if (weights.Length != numWeights)
            {
                throw new Exception("Bad weights array");
            }

            // pointer into weights;
            int k = 0;

            for (int i = 0; i < numInput; ++i)
            {
                for (int j = 0; j < numHidden; ++j)
                {
                    ihWeights[i][j] = weights[k++];
                }
            }

            for (int i = 0; i < numHidden; ++i)
            {
                hBias[i] = weights[k++];
            }

            for (int i = 0; i < numHidden; ++i)
            {
                for (int j = 0; j < numOutput; ++j)
                {
                    ohWeights[i][j] = weights[k++];
                }
            }

            for (int i = 0; i < numOutput; ++i)
            {
                oBiases[i] = weights[k++];
            }
        }
        /// <summary>
        ///  Compute Outpus
        /// </summary>
        /// <param name="xValues"></param>
        /// <returns></returns>
        internal double[] ComputeOutpus(double[] xValues)
        {
            if (xValues.Length != numInput)
            {
                throw new Exception("Bad xValues array");
            }

            double[] hSum = new double[numHidden];
            double[] oSum = new double[numOutput];

            for (int i = 0; i < xValues.Length; ++i)
            {
                inputs[i] = xValues[i];
            }

            for (int j = 0; j < numHidden; ++j)
            {
                for (int i = 0; i < numInput; ++i)
                {
                    hSum[j] += inputs[i] * ihWeights[i][j];
                }
            }

            for (int i = 0; i < numHidden; ++i)
            {
                hSum[i] += hBias[i];
            }

            Console.WriteLine("\nPre-activation hidden sums:");
            NeuronNetwork.Program.ShowVector(hSum, 4, 4, true);

            for (int i = 0; i < numHidden; ++i)
            {
                if (this.hActivation == Activation.HyperTan)
                {
                    hOutpus[i] = HyperTan(hSum[i]);
                }
                else if (this.hActivation == Activation.LogSigmoid)
                {
                    hOutpus[i] = LogSigmoid(hSum[i]);
                }

            }

            Console.WriteLine("\nHidden outputs:");
            NeuronNetwork.Program.ShowVector(hOutpus, 4, 4, true);

            for (int j = 0; j < numOutput; ++j)
            {
                for (int i = 0; i < numHidden; ++i)
                {
                    oSum[j] += hSum[i] * ohWeights[i][j];
                }
            }

            for (int i = 0; i < numOutput; ++i)
            {
                oSum[i] += oBiases[i];
            }

            Console.WriteLine("\nPre-activation output sums:");
            NeuronNetwork.Program.ShowVector(oSum, 2, 4, true);

            double[] softOutput = null;

            if (oActivation == Activation.Softmax)
            {
                softOutput = SoftMax(oSum);
            }

            for (int i = 0; i < outputs.Length; ++i)
            {
                outputs[i] = softOutput[i];
            }

            double[] result = new double[numOutput];
            for (int i = 0; i < outputs.Length; ++i)
            {
                result[i] = outputs[i];
            }

            return result;
        }

        /// <summary>
        /// HyperTan tanh method pre-activation
        /// </summary>
        /// <param name="v"></param>
        /// <returns></returns>
        private static double HyperTan(double v)
        {
            if (v < -20.0)
            {
                return -1.0;
            }
            else if (v > 20.0)
            {
                return 1.0;
            }
            else
            {
                return Math.Tanh(v);
            }
        }

        private static double LogSigmoid(double v)
        {
            if (v < -45.0)
            {
                return 0.0;
            }
            else if (v > 45.0)
            {
                return 1.0;
            }
            else
            {
                return 1.0 / (1.0 + Math.Exp(-v));
            }
        }

        /// <summary>
        /// Make a Matrix for connection between Nodes
        /// </summary>
        /// <param name="rows"></param>
        /// <param name="cols"></param>
        /// <returns></returns>
        private static double[][] MakeMatrix(int rows, int cols)
        {
            double[][] result = new double[rows][];
            for (int i = 0; i < rows; ++i)
            {
                result[i] = new double[cols];
            }
            return result;
        }

        /// <summary>
        /// Method to get Outputs
        /// </summary>
        /// <returns></returns>
        public double[] GetOutputs()
        {
            double[] result = new double[numOutput];
            for (int i = 0; i < numOutput; ++i)
            {
                result[i] = this.outputs[i];
            }

            return result;
        }

        private static double[] SoftMax(double[] oSums)
        {
            // does all output nodes at once 
            // determinate max oSum
            double max = oSums[0];
            for (int i = 0; i < oSums.Length; ++i)
            {
                if (oSums[i] > max)
                {
                    max = oSums[i];
                }
            }

            // determinate scaling factor -- sum of exp(val - max)
            double scale = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
            {
                scale += Math.Exp(oSums[i] - max);
            }


            //get the result
            double[] result = new double[oSums.Length];

            for (int i = 0; i < oSums.Length; ++i)
            {
                result[i] = Math.Exp(oSums[i] - max) / scale;  // exp(val - max)/ scale
            }

            return result;
        }

        public static double[] SoftmaxNAive(double[] oSums)
        {
            double[] result = new double[oSums.Length];
            double denom = 0.0;
            for (int i = 0; i < oSums.Length; ++i)
            {
                denom += Math.Exp(oSums[i]);
            }

            for(int i =0; i <  oSums.Length; ++i)
            {
                result[i] = Math.Exp(oSums[i]) / denom;
            }

            return result;
        }

    }
}
