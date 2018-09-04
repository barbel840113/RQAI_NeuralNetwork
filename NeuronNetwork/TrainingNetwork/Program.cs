using System;
using System.IO;

namespace TrainingNetwork
{
    class TrainingProgram
    {
        private static object trainData;

        static void Main(string[] args)
        {
            Console.WriteLine("\nBegin neural network training demo");

            double[][] allData = new double[150][];
            allData[0] = new double[] { 5.1, 3.5, 1.4, 0.2, 0, 0, 1 };

            //Define remaining data here
            //Create train and test data

            int numInput = 4;
            int numHidden = 7;
            int numOutput = 3;

            NeuralNetwork nn = new NeuralNetwork(numInput, numHidden, numOutput);

            int maxEpochs = 1000;
            double learningRate = 0.05;
            double momentum = 0.01;
            nn.Train(trainData, maxEpochs, learningRate, momentum);

            Console.WriteLine("\nFirst 3 rows of training data: ");
            ShowMatrix(trainData, 3, 1, true);
            Console.WriteLine("First 3 rows of test data:");
            ShowMatrix(testData, 3, 1, true);
        }

        public static double[][] LoadData(string dataFile, int numRows, int numCols)
        {
            double[][] result = new double[numRows][];

            FileStream ifs = new FileStream(dataFile, FileMode.OpenOrCreate);
            StreamReader sfs = new StreamReader(ifs);

            string line = "";
            string[] tokens = null;
            int i = 0;
            while ((line = sfs.ReadLine()) != null)
            {
                tokens = line.Split(",");
                result[i] = new double[numCols];
                for (int j = 0; j < numCols; ++j)
                {
                    result[i][j] = double.Parse(tokens[j]);
                }
                ++i; // increase line
            }

            sfs.Close();
            ifs.Close();

            return result;
        }

        /// <summary>
        /// MakeTrainTest
        /// </summary>
        /// <param name="allData"></param>
        /// <param name="seed"></param>
        /// <param name="trainData"></param>
        /// <param name="testData"></param>
        public static void MakeTrainTest(double[][] allData, int seed, out double[][] trainData, out double[][] testData)
        {
            Random rd = new Random(seed);

            int totRows = allData.Length;
            int numCols = allData[0].Length;
            int trainRows = (int)(totRows * 0.80); // 80% hard-coded 80-20 split
            int testRows = (int)(totRows * 0.20); // 20%
            trainData = new double[trainRows][];
            testData = new double[testRows][];

            double[][] copy = new double[allData.Length][];
            for (int i = 0; i < copy.Length; ++i)
            {
                copy[i] = allData[i]; // all data to copy
            }

            // refernce copy 
            for (int i = 0; i < copy.Length; ++i)
            {
                int r = rd.Next(i, copy.Length);
                double[] tmp = copy[r];
                copy[r] = copy[i];
                copy[i] = tmp;
            }

            for (int i = 0; i < trainRows; ++i)
            {
                trainData[i] = new double[numCols];
                for (int j = 0; j < numCols; ++j)
                {
                    trainData[i][j] = copy[i][j];
                }
            }

            for (int i = 0; i < testRows; ++i)
            {
                testData[i] = new double[numCols];
                for (int j = 0; j < numCols; ++j)
                {
                    testData[i][j] = copy[i + trainRows][j]; // be carefull
                }
            }
        }

        /// <summary>
        /// ShowVector
        /// </summary>
        /// <param name="vector"></param>
        /// <param name="valsPerRow"></param>
        /// <param name="decimals"></param>
        /// <param name="newLine"></param>
        public static void ShowVector(double[] vector, int valsPerRow, int decimals, bool newLine)
        {
            for (int i = 0; i < vector.Length; ++i)
            {
                if (i % 100 == 0)
                {
                    Console.WriteLine(vector[i].ToString("F" + decimals).PadLeft(decimals + 4) + " ");
                }
            }

            if (newLine)
            {
                Console.WriteLine("");
            }
        }

        public static void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
        {

        }


        public void ShowMatrix(double[][] matrix, int numRows, int decimals, bool newLine)
        {
            for (int i = 0; i < numRows; ++i)
            {
                Console.WriteLine(i.ToString().PadLeft(3) + " : ");

                for (int j = 0; j < matrix[i].Length; ++j)
                {
                    if (matrix[i][j] >= 0.0)
                    {
                        Console.WriteLine(" ");
                    }
                    else
                    {
                        Console.WriteLine("-");
                    }
                }

                Console.WriteLine("");
            }

            if (newLine)
            {
                Console.WriteLine("");
            }
        }
    }
}
