using System;
using System.Threading;

namespace BackPropagation
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin back-propagation demo\n");

            // all program control logic goes here
            int numInput = 3;
            int numHidden = 4;
            int numOutput = 2;

            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

            double[] weights = new double[26] {
                0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08,
                0.09, 0.10, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16,
                0.17, 0.18, 0.19, 0.20, 0.21, 0.22, 0.23, 0.24,
                0.25, 0.26
            };

            Console.WriteLine("Setting dummy initial weights to:");
            ShowVector(weights, 8, 2, true);
            nn.SetWeights(weights);

            double[] xValues = new double[3] { 1.0, 2.0, 3.0 }; // Inputs
            double[] tValues = new double[2] { 0.2500, 0.7500 }; // Target output
           
            Console.WriteLine("\nSetting fixed inputs");
            ShowVector(xValues, 3, 1, true);
            Console.WriteLine("Setting fixed target output = ");
            ShowVector(tValues, 2, 4, true);

            double learningRate = 0.05;
            double momentum = 0.01;
            int maxEpochs = 1000;
            Console.WriteLine("\nSettings learning reate = " + learningRate.ToString("F2"));
            Console.WriteLine("Setting  momentum = " + momentum.ToString());
            Console.WriteLine("Setting max epochs = " + maxEpochs + "\n");

            // find wieghts
            nn.FindWeights(tValues, xValues, learningRate, momentum, maxEpochs);

            double [] bestWeights = nn.GetWeights();
            Console.WriteLine("\nBest weights found:");

            ShowVector(bestWeights, 8, 4, true);

            Console.WriteLine("\nEnd back-propagation demo\n");
            Console.ReadLine();
        }

        public static void ShowVector(double[] vector, int valsPerRow, int demicals, bool newLine)
        {
            for(int i = 0; i < vector.Length; ++i)
            {
                if(i > 0 && i % valsPerRow == 0)
                {
                    Console.WriteLine("");
                }
                Console.WriteLine(vector[i].ToString("F" + demicals).PadLeft(demicals + 4) + " ");
            }

            if(newLine == true)
            {
                Console.WriteLine("");
            }
        }

        public static void ShowMatrix(double[][] matrix, int decimals)
        {
            int cols = matrix[0].Length;
            for (int i = 0; i < matrix.Length; ++i)
            {
                ShowVector(matrix[i], cols, decimals, true);
            }
        }
    }
}
