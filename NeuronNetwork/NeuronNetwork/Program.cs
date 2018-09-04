using System;

namespace NeuronNetwork
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin feed-forward neural network.");

            int numInput = 3;
            int numHidden = 4;
            int numOutput = 2;

            Console.WriteLine("Creating a 3-4-2 tanh-softmax neural network");
            NeuralNetworks nn = new NeuralNetworks(numInput, numHidden, numOutput, NeuralNetworks.Activation.HyperTan, NeuralNetworks.Activation.Softmax);

            //set weights
            double[] weights = new double[] {
             0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10,
             0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20,
             0.21, 0.22, 0.23, 0.24, 0.25, 0.26
            };

            Console.WriteLine("\nSetting dummy weights and biases");
            ShowVector(weights, 8, 2, true);
            nn.SetWeights(weights);

            //set input
            double[] xValues = new double[] { 1.0, 2.0, 3.0 };
            Console.WriteLine("\nInputs are: ");
            ShowVector(xValues, 3, 1, true);

            // compute and display output
            Console.WriteLine("\nComputing outputs");
            double[] yValues = nn.ComputeOutpus(xValues);
            Console.WriteLine("\nOutputs computed");
            Console.WriteLine("\nOutputs are:");
            ShowVector(yValues, 2, 4, true);

            Console.WriteLine("\nEnd feed-forward demo\n");
            Console.ReadLine();
        }

        public static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % valsPerRow == 0)
                {
                    Console.WriteLine(" ");
                }
                Console.Write(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
            }
        }
    }
}
