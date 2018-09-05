using System;
using System.Collections.Generic;
using System.Text;

namespace BackPropagation
{
    public class NeuralNetwork
    {
        private int numInput;
        private int numHidden;
        private int numOutput;

        // inputs, inputs weights, hidden biases.
        private double[] inputs;
        private double[][] ihWeights;
        private double[] hOutputs;
        private double[] hBiases;

        // outputs, outputs biases, outputs weights
        private double[] outputs;
        private double[][] ohWeights;
        private double[] oBiases;

        // grade output gradeint for back-propagation
        private double[] oGrads;
        // Hidden gradients for back-propagation.
        private double[] hGrads;
       

        private double[][] ihPreviousWeightsDelta;
        private double[] hPreviuosBiasesDelta;
        private double[][] ohPreviousWeightsDelta;
        private double[] oPreviousBiasesDelta;

        public NeuralNetwork(int numInput, int numHidden, int numOutput)
        {
            this.numHidden = numHidden;
            this.numInput = numInput;
            this.numOutput = numOutput;

            this.inputs = new double[this.numInput];
            this.ihWeights = MakeMatrix(this.numInput, this.numHidden);
            this.hBiases = new double[this.numHidden];
            this.hOutputs = new double[this.numHidden];

            this.outputs = new double[this.numOutput];
            this.oBiases = new double[this.numOutput];
            this.ohWeights = MakeMatrix(this.numHidden, this.numOutput);

            this.hGrads = new double[this.numHidden];
            this.oGrads = new double[this.numOutput];

            ihPreviousWeightsDelta = MakeMatrix(this.numInput, this.numHidden);
            hPreviuosBiasesDelta = new double[this.numHidden];
            ohPreviousWeightsDelta = MakeMatrix(this.numHidden, this.numOutput);
            oPreviousBiasesDelta = new double[this.numOutput];

            InitMatrix(this.ihPreviousWeightsDelta, 0.011);
            InitVector(this.hPreviuosBiasesDelta, 0.011);
            InitMatrix(this.ohPreviousWeightsDelta, 0.011);
            InitVector(this.oPreviousBiasesDelta, 0.011);

        }


        internal void FindWeights(double[] tValues, double[] xValues, double learningRate, double momentum, int maxEpochs)
        {
            int epoch = 0;
            while(epoch < maxEpochs)
            {
               
                double[] yValues = this.ComputeOutputs(xValues);
                this.Updateweights(tValues, learningRate, momentum);

                if(epoch % 100 == 0)
                {
                    Console.WriteLine("epoch = " + epoch.ToString().PadLeft(5) + "    curr outputs = ");
                    Program.ShowVector(yValues, 2, 4, true);
                }

                ++epoch; // find loop
            }
        }

        private void Updateweights(double[] tValues, double learningRate, double momentum)
        {
            if(tValues.Length != numOutput)
            {
                throw new Exception("target array not same length as output in UpdateWeights");
            }

            // method computes output gradients
            for(int i =0; i < oGrads.Length; ++i)
            {
                double derivate = (1 - outputs[i]) * outputs[i];
                oGrads[i] = derivate * (tValues[i] - outputs[i]);
            }

            // calculate hidden gradient
            for(int i  =0; i < hGrads.Length; ++i)
            {
                double derivate = (1 - hOutputs[i]) * (1 + hOutputs[i]);
                double sum = 0.0;
                for(int j = 0; j < numOutput; ++j)
                {
                    sum += oGrads[j] * ohWeights[i][j];
                }
                hGrads[i] = derivate * sum;
            }

            // update weights for input-hidden network
            for(int i =0; i < ihWeights.Length; ++i)
            {
                for(int j = 0; j < ihWeights[i].Length; ++j)
                {
                    double delta = learningRate * hGrads[j] * inputs[i];
                    ihWeights[i][j] += delta;
                    ihWeights[i][j] += momentum * ihPreviousWeightsDelta[i][j];
                    ihPreviousWeightsDelta[i][j] = delta; // save the delta
                }
            }

            for(int i = 0; i < numHidden; ++i)
            {
                double delta = learningRate * hGrads[i];
                hBiases[i] += delta;
                hBiases[i] += momentum * hPreviuosBiasesDelta[i];
                hPreviuosBiasesDelta[i] = delta; // save delta  * 1.0 // for normalization                            
            }

            // method update hidden output
            for (int i = 0; i < ohWeights.Length; ++i)
            {
                for (int j = 0; j < ohWeights[j].Length; ++j)
                {
                    double delta = learningRate * oGrads[j] * outputs[j];
                    ohWeights[i][j] += delta;
                    ohWeights[i][j] = momentum * ohPreviousWeightsDelta[i][j];
                    ohPreviousWeightsDelta[i][j] = delta; //save delta
                }
            }

            // update output biases
            for(int i =0; i < oBiases.Length; ++i)
            {
                double delta = learningRate * oGrads[i] * 1.0;
                oBiases[i] += delta;
                oBiases[i] = momentum * oPreviousBiasesDelta[i];
                oPreviousBiasesDelta[i] = delta;
            }

            //update weights
        }

        /// <summary>
        /// INitialize Vector
        /// </summary>
        /// <param name="hPreviuosBiasesDelta"></param>
        /// <param name="v"></param>
        private static void InitVector(double[] hPreviuosBiasesDelta, double v)
        {
            for (int i = 0; i < hPreviuosBiasesDelta.Length; ++i)
            {
                hPreviuosBiasesDelta[i] = v;
            }
        }

        /// <summary>
        /// Initialize Matrix
        /// </summary>
        /// <param name="ihPreviousWeightsDelta"></param>
        /// <param name="v"></param>
        private static void InitMatrix(double[][] ihPreviousWeightsDelta, double v)
        {
            int rows = ihPreviousWeightsDelta.Length;
            int cols = ihPreviousWeightsDelta[0].Length;
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    ihPreviousWeightsDelta[i][j] = v;
                }
            }
        }

        private double[] ComputeOutputs(double[] xValues)
        {
            double[] hSum = new double[numHidden];
            double[] oSum = new double[numOutput];

            //setup weights
            for (int i = 0; i < xValues.Length; ++i)
            {
                this.inputs[i] = xValues[i];
            }

            // count sum (inputI * ihWeightI)
            for (int i = 0; i < numHidden; ++i)
            {
                for (int j = 0; j < numInput; ++j)
                {
                    hSum[i] += inputs[j] * ihWeights[j][i];
                }
            }

            //add biases
            for (int i = 0; i < numHidden; ++i)
            {
                hSum[i] += hBiases[i];
            }

            // Pre-Activation function
            for (int i = 0; i < numHidden; ++i)
            {
                hOutputs[i] = HyperTan(hSum[i]);
            }

            // calculate sum ofr activation method hOutput[I] * outputWeights[I][J]
            for (int i = 0; i < numOutput; ++i)
            {
                for (int j = 0; j < numHidden; ++j)
                {
                    oSum[i] += hOutputs[j] * ohWeights[j][i];
                }
            }

            // add outputBiases + output
            for (int i = 0; i < numOutput; ++i)
            {
                oSum[i] += oBiases[i];
            }

            double[] softOutput = SoftMax(oSum);
            for (int i = 0; i < outputs.Length; ++i) // add softmad does all output once
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
        /// Hyper Tangolic function
        /// </summary>
        /// <param name="oSum"></param>
        /// <returns></returns>
        private static double[] SoftMax(double[] oSum)
        {
            double max = oSum[0];
            for (int i = 0; i < oSum.Length; ++i)
            {
                if (oSum[i] > max)
                {
                    max = oSum[i];
                }
            }

            double scale = 0.0;
            for(int i =0; i < oSum.Length; ++i)
            {
                scale += Math.Exp(oSum[i] - max);
            }

            double[] result = new double[oSum.Length];
            for(int i = 0; i < oSum.Length; ++i)
            {
                result[i] = Math.Exp(oSum[i] - max) / scale;
            }

            return result;
        }

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

        public void SetWeights(double[] weights)
        {
            int numWeights = (this.numInput * this.numHidden) // input hidden weights
            + this.numHidden // input hidden biases
            + (this.numHidden * this.numOutput)  // output hidden weights
            + this.numOutput; // output biases

            if (weights.Length != numWeights)
            {
                throw new Exception("The Weights are not correct");
            }

            int k = 0; // pointer to wieghts
            // input weights or hidden weights
            for (int i = 0; i < this.numInput; ++i)
            {
                for (int j = 0; j < this.numHidden; ++j)
                {
                    this.ihWeights[i][j] = weights[k++];
                }
            }

            // hidden biases == number of hidden Nodes
            for (int i = 0; i < this.numHidden; ++i)
            {
                this.hBiases[i] = weights[k++];
            }

            // for output hidden weights
            for (int i = 0; i < this.numHidden; ++i)
            {
                for (int j = 0; j < this.numOutput; ++j)
                {
                    this.ohWeights[i][j] = weights[k++];
                }
            }

            // for output biases
            for (int i = 0; i < this.numOutput; ++i)
            {
                this.oBiases[i] = weights[k++];
            }

        }

        public double[] GetWeights()
        {
            int numweights = (this.numInput * this.numHidden) + this.numHidden + (this.numHidden * this.numOutput) + this.numOutput;

            double[] result = new double[numweights];
            int k = 0; // Pointer to result

            // add hWeights into result
            for (int i = 0; i < this.numInput; ++i)
            {
                for (int j = 0; j < this.numHidden; ++j)
                {
                    result[k++] = this.ihWeights[i][j];
                }
            }

            // add hBiases into result
            for (int i = 0; i < this.numHidden; ++i)
            {
                result[k++] = this.hBiases[i];
            }

            // add outputWeights into result
            for (int i = 0; i < this.numHidden; ++i)
            {
                for (int j = 0; j < this.numOutput; ++j)
                {
                    result[k++] = ohWeights[i][j];
                }
            }
            // add outputBiases into result
            for (int i = 0; i < this.numOutput; ++i)
            {
                result[k++] = this.oBiases[i];
            }
            return result;
        }

        public void SetWeights(double[][] ihWeights, double[] hBiases, double[][] ohWeights, double[] oBiases)
        {

            int k = 0; // pointer to wieghts
            // input weights or hidden weights
            for (int i = 0; i < this.numInput; ++i)
            {
                for (int j = 0; j < this.numHidden; ++j)
                {
                    this.ihWeights[i][j] = ihWeights[i][j];
                }
            }

            if (this.numHidden != hBiases.Length)
            {
                throw new Exception("The Array of Input hidden biases error");
            }

            // hidden biases == number of hidden Nodes
            for (int i = 0; i < this.numHidden; ++i)
            {
                this.hBiases[i] = hBiases[i];
            }

            // for output hidden weights
            for (int i = 0; i < this.numHidden; ++i)
            {
                for (int j = 0; j < this.numOutput; ++j)
                {
                    this.ohWeights[i][j] = ohWeights[i][j];
                }
            }

            if (this.numOutput != oBiases.Length)
            {
                throw new Exception("Output Biases Array Length error");
            }
            // for output biases
            for (int i = 0; i < this.numOutput; ++i)
            {
                this.oBiases[i] = oBiases[i];
            }

        }


        /// <summary>
        /// Make Matrix
        /// </summary>
        /// <param name="rows">Number of Rows</param>
        /// <param name="cols">Number of Cols</param>
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
    }
}
