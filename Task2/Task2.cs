using System;
using System.Collections.Generic;

namespace ИТИБ2
{

    class NeuralNetwork
    {
        /// <summary>
        /// Конструктор класса Нейронная сеть 
        /// </summary>
        /// <param name="p_">Размер окна</param>
        /// <param name="a_">Левая граница отрезка</param>
        /// <param name="b_">Правая граница отрезка</param>
        /// <param name="N_">Количество точек</param>
        /// <param name="TeachingRate_">Норма обучения</param>
        public NeuralNetwork(int p_, double a_, double b_, int N_, double TeachingRate_)
        {
            p = p_; a = a_; b = b_; N = N_; TeachingRate = TeachingRate_;

            weights = new double[p+1];
            deltaWeights = new double[p+1];
            SubsequentInterval();
            ComputingDiscreteSets();
            TotalError();
            Print();
            
        }

        /// <summary>
        /// Печатание каждой эпохи
        /// </summary>
        private void Print()
        {
            Console.Write("Эпоха: ");
            Console.Write("{0,4}", $"{Epoch}");
            Console.Write(" | Вектор весов: ");
            foreach (double i in weights)
            {
                Console.Write("{0,7}", $"{Math.Round(i, 3).ToString("N3")} ");
            }
            Console.Write($" | Суммар. ошибка: ");
            Console.Write("{0,2}", $"{Error}\n");
        }
        
        /// <summary>
        /// Обучение + прогноз нейронной сети
        /// </summary>
        /// <param name="MaxEpoch"> Максимальное количество эпох</param>
        public void Life(int MaxEpoch)
        {
            while (Epoch < MaxEpoch)
            {
                SlidingWindow();
                TotalError();
                ++Epoch;
                Print();
            };
            Forecasting();
        }

        /// <summary>
        /// Прогнозирование нейронной сети
        /// </summary>
        private void Forecasting()
        {
            for (int i = (N - p - 1); i <= (2*N - p - 2); ++i)
            {
                var PredictedValue = CalculationX_t(i);

                if (i == N - p - 1) // замена значения функции в точке конца отрезка обучения и начала отрезка прогнозирования
                    xt[N - 1] = PredictedValue;
                else
                    xt.Add(PredictedValue);
                Forecast.Add(new KeyValuePair<double, double>(PredictedValue, x_coordinates[i+p]));
            }
            Console.Write($"\nПрогноз на интервале [{b}, {c}]:");

            for (var i = 0; i < Forecast.Count; ++i)
            {
                Console.Write("\n(");
                Console.Write("{0,16}", Forecast[i].Value.ToString("N14"));
                Console.Write("; ");
                Console.Write("{0,16}", Forecast[i].Key.ToString("N14"));
                Console.Write(")");
            }
            Console.WriteLine();
            Forecast.Clear();
        }

        /// <summary>
        /// Метод "скользящего окна"
        /// </summary>
        private void SlidingWindow()
        {
            for(int i = 0; i <= (N - p - 1); ++i)
            {
                WidrowHoff(i);
            }
        }

        /// <summary> Пересчитывания весов для каждого набора Х</summary>
        /// /// <param name="index">Индекс Х, c которого начинается окно</param>
        private void WidrowHoff(int index)
        {
            double x_t = CalculationX_t(index);
            for (int j = 1; j <= p; ++j)
            {
                deltaWeights[j] = TeachingRate * (xt[index + p] - x_t) * xt[index + j -1];
                weights[j] += deltaWeights[j];
            }
          
        }

        /// <summary> Расчет последующего интервала</summary>
        private void SubsequentInterval()
        {
            c = 2 * b - a;
        }

        ~NeuralNetwork() { }

        /// <summary> Так как расположение точек равномерное, то следует 
        /// расчитать разность арифметической прогрессии</summary>
        /// <param name="max">Правая граница отрезка</param>
        /// <param name="min">Левая граница отрезка</param>
        private void СalculateDifferenceProgress(double max, double min)
        {
            DifferenceProgress = Math.Sqrt((max - min) * (max - min)) / (2*N-2.0);
        }

        /// <summary> Высчитывание дискретных наборов значений функции x(t)</summary>
        private void ComputingDiscreteSets()
        {
            СalculateDifferenceProgress(c, a);
            double t = a;
            for(double i = 0; i < 2*N - 1; ++i)
            {
                if( i <= N - 1 ) // координаты х нужны на весь отрезок [a,c], а координаты у только на [a,b]
                    xt.Add(X_tFunction(t));
                x_coordinates.Add(t);
                t += DifferenceProgress;
            }
        }

        /// <summary> Функция x(t)</summary>
        /// <param name="t">Момент времени</param>
        private double X_tFunction(double t)
        {
            return t * t * Math.Sin(t);
        }

        /// <summary> Высчитывание X~(t)</summary>
        /// <param name="index">Индекс Х, c которого начинается окно</param>
        private double CalculationX_t(int index)
        {
            double x_t = 0;
            for (int k = 1; k <= p; ++k)
            {
                x_t += (weights[k]*xt[index + k-1]);
            }
            return x_t + weights[0];
        }

        /// <summary>
        /// Расчет суммарной квадратичной ошибки в конце эпохи
        /// </summary>
        private void TotalError()
        {
            Error = 0;
            for (int i = 0; i <= (N - p - 1); ++i)
            {
                double delta = xt[i + p] - CalculationX_t(i);
                Error += delta * delta;
            }

            Error = Math.Sqrt(Error);
        }

        private int p; // длина окна
        private double a, b; // границы временного интервала
        private double c; // граница справа последующего интервала
        private int N; // количество точек 
        private List <double> xt = new List<double>(); // дискретный набор значений функции
        // Спрогнозированные значения функции + координата по по оси x
        private List<KeyValuePair<double, double>> Forecast = new List<KeyValuePair<double, double>>(); 
        private List<double> x_coordinates  = new List<double>();
        private double[] weights; // Веса
        private double TeachingRate; // Норма обучения
        private double DifferenceProgress; // Разность арифметической погрессии (расположение точек на отрезке равномерное)
        private int Epoch = 0; // Номер эпохи
        private double[] deltaWeights; // Дельты весов
        private double Error; // Cуммарная квадратичная ошибка в конце эпохи
    }

        class Program
    {
        static void Main(string[] args)
        {
            NeuralNetwork my = new NeuralNetwork(9, -1, 1, 20, 1);
            my.Life(5000);
        }
    }
}
