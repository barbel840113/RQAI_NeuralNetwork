using System;
using System.Threading;

namespace ThreadTutorial
{
    class Program
    {
        public static object readOnly = new object();
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            for(int i =0; i< 100; i++)
            {
                lock(readOnly)
                {
                    new Thread(() => {
                        Console.Write(i + " ");
                    }).Start();
                }
               
            }

            Console.ReadLine();
        }
    }
}
