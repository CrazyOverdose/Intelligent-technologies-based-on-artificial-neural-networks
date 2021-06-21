using System;
using System.Collections.Generic;
using System.Linq;

namespace Нейронки7_8_
{
    class RNSHopfield
    {
        /// <summary>
        /// Конструктор рекуррентной нейронной сети Хопфилда
        /// </summary>
        /// <param name="x_"> Список запоминаемых векторов</param>
        public RNSHopfield(List<int[,]> x_)
        {
            x = x_;

            transposed_x = new List<int[,]>(x.Count);

            for (int i = 0; i < x.Count(); ++i)
                Transposition_x(i);

            CalculationW();
            PrintMatrix(w);
        }


        /// <summary>
        /// Печать результата
        /// </summary>
        private void Print()
        {
            for (int i = 0; i < 5; ++i)
            {
                for (int j = i; j < Newy.Length; j += 5)
                {
                    if (Newy[j] == 1)
                        Console.Write("$");
                    else
                        Console.Write(" ");
                }

                Console.WriteLine();
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Проверка массивов на равенство
        /// </summary>
        /// <param name="arr1">Первый массив</param>
        /// <param name="arr2">Второй массив</param>
        /// <returns>Результат сравнения: true - массивы равны, false - массивы не равны</returns>
        public static bool AreEqual(int[] arr1, int[] arr2)
        {

            int n = arr1.Length;
            int m = arr2.Length;

            if (n != m)
                return false;

            for (int i = 0; i < n; i++)
                if (arr1[i] != arr2[i])
                    return false;

            return true;
        }

        /// <summary>
        /// Рабочий режим
        /// </summary>
        /// <param name="x"> Вектор, подаваемый на вход нейронной сети</param>
        public void WorkMode(int[] x)
        {
            Newy = x;
            Oldy = new int[x.Length];
            net = new int[x.Length];

            Console.WriteLine("Вход нейронной сети:\n");
            Print();

            while (true)
            {

                if (AreEqual(Oldy, Newy))
                    break;

                for (int i = 0; i < Oldy.Length; ++i)
                    Oldy[i] = Newy[i];

                CalculationNewY();
            }
            Console.WriteLine("Результат работы нейронной сети:\n");
            Print();
        }

        /// <summary>
        /// Расчет отклика
        /// </summary>
        private void CalculationNewY()
        {
            for (int i = 0; i < Oldy.Length; ++i)
            {
                CalculationNet(i);
                CalculationY(i);
            }

        }
        /// <summary>
        /// Расчет net в асинхронном режиме работы
        /// </summary>
        /// <param name="k">Индекс расчитываемого net</param>
        private void CalculationNet(int k)
        {
            int FirstPart = 0, SecondPart = 0;
            for (int j = 0; j <= k - 1; ++j)
                FirstPart += w[j, k] * Newy[j];

            for (int j = k + 1; j < Oldy.Length; ++j)
                SecondPart += w[j, k] * Oldy[j];

            net[k] = SecondPart + FirstPart;
        }

        /// <summary>
        /// Расчет yk в асинхронном режиме работы
        /// </summary>
        /// <param name="k">Индекс расчитываемого yk</param>
        private void CalculationY(int k)
        {
            if (net[k] > 0)
                Newy[k] = 1;
            else if (net[k] < 0)
                Newy[k] = -1;
            else
                Newy[k] = Oldy[k];
        }


        /// <summary>
        /// Получение транспонированных векторов
        /// </summary>
        /// <param name="k">Индекс вектора для транспонирования</param>
        private void Transposition_x(int k)
        {
            var Transposition = new int[x[k].GetLength(1), x[k].GetLength(0)];

            for (int i = 0; i < x[k].GetLength(1); i++)
            {
                for (int j = 0; j < x[k].GetLength(0); j++)
                {
                    Transposition[i, j] = x[k][j, i];
                }

            }
            transposed_x.Add(Transposition);
        }

        /// <summary>
        /// Расчет весов Хопфилда
        /// </summary>
        private void CalculationW()
        {
            // Умножение транспонированного запоминаемого векторо на запоминаемый вектор
            var Mult = new List<int[,]>();
            for (int i = 0; i < x.Count(); ++i)
                Mult.Add(MatrixMultiplication(x[i], transposed_x[i]));

            // Сложение результатов умножения для всех запоминаемых векторов
            var Res = Mult[0];
            for (var i = 1; i < Mult.Count; ++i)
            {
                Res = MatrixSum(Res, Mult[i]);
            }

            w = Res;

            // Обнуление главной диагонали
            for (var i = 0; i < w.GetUpperBound(0) + 1; i++)
            {
                for (var j = 0; j < w.GetUpperBound(1) + 1; j++)
                {
                    w[i, j] = (i == j) ? 0 : w[i, j];
                }
            }
        }

        /// <summary>
        /// Умножние матриц
        /// </summary>
        /// <returns>Матрица - результат умножения</returns>
        private int[,] MatrixMultiplication(int[,] matrixA, int[,] matrixB)
        {

            var matrixC = new int[matrixA.GetUpperBound(0) + 1, matrixB.GetUpperBound(1) + 1];

            for (var i = 0; i < matrixA.GetUpperBound(0) + 1; i++)
            {
                for (var j = 0; j < matrixB.GetUpperBound(1) + 1; j++)
                {
                    matrixC[i, j] = 0;

                    for (var k = 0; k < matrixA.GetUpperBound(1) + 1; k++)
                    {
                        matrixC[i, j] += matrixA[i, k] * matrixB[k, j];
                    }
                }
            }

            return matrixC;
        }

        ~RNSHopfield() { }

        /// <summary>
        /// Функция печати матрицы
        /// </summary>
        /// <param name="matrix">Матрица, требующая печати</param>
        private void PrintMatrix(int[,] matrix)
        {
            Console.WriteLine("Матрица весов:\n");
            for (var i = 0; i < matrix.GetUpperBound(0) + 1; i++)
            {
                for (var j = 0; j < matrix.GetUpperBound(1) + 1; j++)
                {
                    Console.Write(matrix[i, j].ToString().PadLeft(4));
                }

                Console.WriteLine();
            }
            Console.WriteLine();
        }

        /// <summary>
        /// Сложение матриц
        /// </summary>
        /// <returns>Результат сложения матриц</returns>
        private int[,] MatrixSum(int[,] matrixA, int[,] matrixB)
        {
            var matrixC = new int[matrixA.GetLength(0), matrixB.GetLength(1)];

            for (var i = 0; i < matrixA.GetLength(0); i++)
            {
                for (var j = 0; j < matrixB.GetLength(1); j++)
                {
                    matrixC[i, j] = matrixA[i, j] + matrixB[i, j];
                }
            }

            return matrixC;
        }

        private List<int[,]> x;
        private List<int[,]> transposed_x;
        private int[] Oldy;
        private int[] Newy;
        private int[] net;
        private int[,] w;
    }
    class Program
    {
        static void Main(string[] args)
        {
            //Список запоминаемых векторов
            var MemorizedVectors = new List<int[,]> {
                    new int[15, 1] { {1 }  ,  {1 }   ,  {1 }  ,  {1 }  ,  {1 }  ,  {1 }  ,  {-1 }  ,  {-1 }  , // 0
                     { -1 }  ,  {1 }  ,  {1 }  ,  {1 } ,  {1 }  ,  {1 }  ,  { 1 } },
                        new int[15, 1] {  { 1 } ,  {-1 }  ,  {1 } ,  {1 } ,  {1 } ,  {1 } ,  {-1 } ,  {1 } , // 2
                     { -1 }  ,  {1 }  ,  {1 }  ,  {1 }  ,  {1 }  ,  {-1 }  ,  {1 } },
                        new int[15, 1] { { 1 }, { -1 }, { -1 }, { 1 }, { 1 }, { 1 }, { -1 }, { 1 },  // 7
                        { -1 }, { -1 }, { 1 }, { 1 }, { -1 }, { -1 }, {-1} },
                };

            RNSHopfield my = new RNSHopfield(MemorizedVectors);

            //Подача рабочих векторов на вход
            Console.WriteLine("___Подача рабочих векторов на вход___\n");
            my.WorkMode(new int[15] { 1  ,  1   ,  1  ,  1  ,  1  ,  1  ,  -1  ,  -1  , // вертикальные 0 2 7 
                     -1  ,  1  ,  1  ,  1 ,  1  ,  1  ,  1  });
            my.WorkMode(new int[15] {   1 ,  -1  ,  1 ,  1 ,  1 ,  1 ,  -1 ,  1 ,
                     -1  ,  1  ,  1  ,  1  ,  1  ,  -1  ,  1 });
            my.WorkMode(new int[15] { 1, -1, -1, 1, 1, 1, -1, 1, -1, -1, 1, 1, -1, -1, -1 });


            //Подача искаженных векторов на вход
            Console.WriteLine("___Подача искаженных векторов на вход___\n");
            my.WorkMode(new int[15] { -1  ,  -1   ,  1  ,  1  ,  1  ,  1  ,  -1  ,  -1  , // вертикальные 0 2 7 
                     -1  ,  1  ,  1  ,  1 ,  1  ,  1  ,  1  });
            my.WorkMode(new int[15] {   -1 ,  -1  ,  1 ,  1 ,  1 ,  1 ,  -1 ,  1 ,
                     -1  ,  1  ,  1  ,  1  ,  1  ,  -1  ,  -1 });
            my.WorkMode(new int[15] { 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, -1, -1, -1 });
        }

    }
}
