using System;

namespace ИТИБ2
{

    class BackPropagation
    {
        /// <summary>
        /// Конструктор класса Алгоритма обратного распространения ошибки
        /// </summary>
        /// <param name="N_,J_, M_">Архитектура</param>
        /// <param name="x1_">Входной вектор</param>
        /// <param name="t_">Целевой вектор</param>
        /// <param name="n_">Норма обучения</param>
        /// <param name="e_">Погрешность</param>
        public BackPropagation(int N_, int J_, int M_, double[] x1_, double[] t_, double n_, double e_)
        {
            N = N_; J = J_; M = M_; x1 = x1_; t = t_; n = n_; e = e_;

            net1 = new double[J + 1];
            net2 = new double[M + 1];
            x2 = new double[J + 1];

            for (int i = 0; i < x2.Length; ++i)
                x2[i] = 1;

            y = new double[M + 1];
            w1 = new double[N + 1,J + 1];
            w2 = new double[J + 1,M + 1];
            delta1 = new double[J + 1];
            delta2 = new double[M + 1];
            TotalError();
        }

        public void Life()
        {   while (Error >= e)
            {
                Error = 0.0;
                FirstStage();
                SecondStage();
                ThirdStage();
                TotalError();
                Print();
                ++Epoch;
            }
        }

        /// <summary>
        /// Печать результата
        /// </summary>
        private void Print()
        {         
                Console.Write("Эпоха: ");
                Console.Write("{0,3}", $"{Epoch}");
                Console.Write($" | Вых. вектор у: ");
                for (var  i = 1; i < y.Length; ++i)
                {
                    Console.Write("{0,7}", $"{Math.Round(y[i], 5).ToString("N5")} ");
                }
                Console.Write($"| Цел. вектор t = ");
                for (var i = 1; i < t.Length; ++i)
                {
                    Console.Write("{0,3}", $"{Math.Round(t[i], 1).ToString("N1")} ");
                }
                Console.Write($"| Сум. ошибка: ");
                Console.Write("{0,4}", $"{Math.Round(Error, 5).ToString("N5") }");
                Console.Write(" | Веса нейронов скрыт. и выход. слоев: w(1) = ");

                for (var j = 1; j <= J; ++j)
                {
                    for (var i = 0; i <= N; ++i)
                    {
                        Console.Write("{0,4}", $"{Math.Round(w1[i,j], 2).ToString("N2")} ");
                    }
                }

                Console.Write("w(2) = ");
                for (var j = 0; j <= J; ++j)
                {
                    for (var m = 1; m <= M; ++m)
                    {
                        Console.Write("{0,5}", $"{Math.Round(w2[j, m], 5).ToString("N5")} ");
                    }
                }
            
                Console.Write("\n\n");
        }

        /// <summary>
        /// Расчет функции активации нейронов скрытого и выходного слоев
        /// </summary>
        private double f_net(double net)
        {
            return (1.0 - Math.Exp(-1.0 * net)) / (1.0 + Math.Exp(-1.0 * net));
        }

        /// <summary>
        /// Расчет производной функции активации нейронов скрытого и выходного слоев
        /// </summary>
        private double df_net(double net)
        {
            return 0.5* (1 - f_net(net)* f_net(net));
        }

        /// <summary>
        /// Расчет комбинированного входа нейронов скрытого слоя
        /// </summary>
        private void CalculationNet1()
        {
            for (var j = 1; j <= J; ++j)
            {
                var net = 0.0;
                for (var i = 1; i <= N; ++i)
                {
                    net += w1[i, j] * x1[i];
                }
                net1[j] = net + w1[0, j];
            }
        }

        /// <summary>
        /// Расчет комбинированного входа нейронов выходного слоя
        /// </summary>
        private void CalculationNet2()
        {
            for (var m = 1; m <= M; ++m)
            {
                var net = 0.0;
                for (var j = 1; j <= J; ++j)
                {
                    net += w2[j, m] * x2[j];
                }
                net2[m] = net +  w2[0, m];
            }
        }

        /// <summary>
        /// Расчет входного сигнала нейронов выходного слоя
        /// </summary>
        private void CalculationX2()
        {
            for (var j = 1; j <= J; ++j)
            {
                x2[j] = f_net(net1[j]);
            }
        }

        /// <summary>
        /// Расчет выхода многослойной нейронной сети 
        /// </summary>
        private void CalculationYm()
        {
            for(var m = 1; m <= M; ++m )
            {
                y[m] = f_net(net2[m]);
            }
        }

        private void FirstStage()
        {
            CalculationNet1();
            CalculationX2();
            CalculationNet2();
            CalculationYm();
        }

        /// <summary>
        /// Расчет ошибки выходного слоя
        /// </summary>
        private void Calculationdelta()
        {
            for (var m = 1; m <= M; ++m)
            {
                delta2[m] = df_net(net2[m])*(t[m] - y[m]);
            }
        }

        /// <summary>
        /// Расчет ошибки скрытого слоя
        /// </summary>
        private void Calculationdelta1()
        {
            for (var j = 1; j <= J; ++j)
            {
                var sum = 0.0;
                for (var m = 1; m <= M; ++m)
                {
                    sum += w2[j, m] * delta2[m];
                }

                delta1[j] = sum * df_net(net1[j]);
            }
        }

        private void SecondStage()
        {
            Calculationdelta();
            Calculationdelta1();
        }

       /// <summary>
       /// Настройка весов нейронов скрытого слоя
       /// </summary>
        private void Calculationw1()
        {
            for (var j = 1; j <= J; ++j)
            {
                var deltaw = 0.0;
                for (var i = 0; i <= N; ++i)
                {
                    deltaw = n * x1[i]*delta1[j];
                    w1[i,j] += deltaw;
                }
            }
        }

        /// <summary>
        /// Настройка весов нейронов выходного слоя
        /// </summary>
        private void Calculationw2()
        {
            for (var j = 0; j <= J; ++j)
            {
                var deltaw = 0.0;
                for (var m = 1; m <= M; ++m)
                {
                    deltaw = n * x2[j] * delta2[m];
                    w2[j, m] += deltaw;
                }
            }
        }

        private void ThirdStage()
        {
            Calculationw1();
            Calculationw2();
        }

        /// <summary>
        ///Расчет суммарной среднеквадратичной ошибки
        /// </summary>
        private void TotalError()
        {
            for (var j = 1; j <= M; ++j)
            {
                Error += (t[j] - y[j])*(t[j] - y[j]); 
            }

            Error = Math.Sqrt(Error);
        }
        

        private readonly int N, J, M; // N-J-M - архитектура
        private readonly double n; //Норма обучения
        private double Error = 0.0; // Cуммарная среднеквадратичная ошибка
        private int Epoch = 0; //Номер эпохи
        private double e; //Погрешность
        private double[] t; //Целевой вектор
        private double[] x2; //Входной сигнал нейронов выходного слоя
        private double[] net1, net2; //Комбинированные входы нейронов скрытого слоя
        private double[] y; //Выход многослойной нейронной сети 
        private double[] delta2, delta1; // Ошибка скрытого и выходного слоев
        private double[] x1; //Входной вектор
        private double[,] w1, w2; //Веса нейронов скрытого и выходного слоев
    }

    class Program
    {
        static void Main(string[] args)
        {
            //в массиве t всегда первый элемент = 0
            BackPropagation my = new BackPropagation(1, 1, 3, new[] { 1.0, -1.0}, new[] {0.0, -1.0, 2.0, 2.0 }, 1, 1.42);
            my.Life();
        }
    }
}
