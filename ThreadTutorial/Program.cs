using System;
using System.Threading;
using System.Diagnostics;

namespace ThreadTutorial
{
    class Program
    {
        public static object readOnly = new object();
        public static int readOnlyInt = 0;
        public static Stopwatch stopWatch = new Stopwatch();
        public static int intePointer = 0;

        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            stopWatch.Start();
            for (intePointer = 0; intePointer < 100; intePointer++)
            {

                new Thread(Startthread).Start();
            }

            
            Console.ReadLine();
        }
           

        public static void Startthread()
        {
            lock (readOnly)
            {
                Console.Write(intePointer + " ");
            }
            
        }
    }
}
