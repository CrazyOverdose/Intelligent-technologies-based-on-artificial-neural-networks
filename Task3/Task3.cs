using System;
using System.Collections.Generic;

namespace ITIB4
{
    class RBF
    {
        /// <summary>
        /// Конструктор класса НС с радиальными базисными функциями 
        /// </summary>
        /// <param name="_teachingRate">Норма обучения</param>
        /// <param name="_sets">Минимальный набор х, используемых для обучения</param>
        /// <param name="flag_">Выбор ФА</param>
        public RBF(double _teachingRate, List<int>[] _sets, bool flag_)
        {
            teachingRate = _teachingRate; sets_x = _sets; flag = flag_;
            CalculationC();
            Calculationt();

            weights = new double[C.Count + 1];
            
            deltaWeights = new double[C.Count + 1];
            fi = new double[C.Count+1];
            fi[0] = 1;

            CalculationALLFA();
            TotalError();
        }

        /// <summary>
        /// Печать результат
        /// </summary>
        private void Print()
        {
            Console.Write("Эпоха: ");
            Console.Write("{0,2}", $"{Epoch}");
            Console.Write(" | Вектор весов: ");
            foreach (double i in weights)
            {
                Console.Write("{0,7}", $"{Math.Round(i, 3).ToString("N3")} ");
            }
            Console.Write($" | Вых. вектор у: {string.Concat(y)} | Цел. вектор t: {string.Concat(t)} | Суммар. ошибка: ");
            Console.Write("{0,2}", $"{Error}\n");
        }

        public void Life()
        {
            while (true)
            {
                Print();

                if (Error == 0)
                    break;

                WidrowHoff();
                ++Epoch;
                CalculationALLFA();
                TotalError();
            }
        }

        ~RBF() { }

        /// <summary>
        /// Расчет целевого вектора
        /// </summary>
        private void Calculationt()
        {
            for (var i = 0; i < t.Length; ++i)
            {
                t[i] = CalculationBF(AllSets[i]) ? 1 : 0;
            }
        }

        /// <summary>
        /// Расчет моделируемая ФА
        /// </summary>
        /// <param name="x">Набор, для которого подсчитывается ФА</param>
        /// <returns></returns>
        private bool CalculationBF(List<int> x)
        {
           return !(!Convert.ToBoolean(x[2]) & Convert.ToBoolean(x[3]) & (!Convert.ToBoolean(x[0]) | Convert.ToBoolean(x[1])));
        }

        /// <summary>
        ///Расчет Центры RBF-нейронов
        /// </summary>
        private void CalculationC()
        {
            List<List<int>> C1 = new List<List<int>>(), C0 = new List<List<int>>();

            foreach (var i in AllSets)
            {
                if (CalculationBF(i))
                {
                    C1.Add(i);
                }
                else
                    C0.Add(i);
            }

            if (C1.Count > C0.Count)
                C = C0;
            else
                C = C1;
        }

        /// <summary> Высчитывание тотальной ошибки</summary>
        private void TotalError()
        {
            int error = 0;
            for (int i = 0; i < t.Length; ++i)
            {
                error += (y[i] == t[i]) ? 0 : 1;
            }

            Error = error;
        }

        /// <summary>
        /// Подсчет выходов RBF-нейронов 
        /// </summary>
        /// <param name="set">Набор, для которого подсчитывается выход RBF-нейронов</param>
        private void Calculationfi(List<int> set)
        {
            for (int j = 1; j < fi.Length; ++j)
            {
                var fi_ = 0;

                for (int i = 0; i < 4; ++i)
                {
                    fi_ += (set[i] - C[j-1][i]) * (set[i] - C[j-1][i]);
                }

                fi[j] = Math.Exp(-1 * fi_);
            }
        }


        /// <summary> Высчитывание net</summary>
        /// <param name="j">Индекс  набора Х, для которого нужно подсчитать net</param>
        private double СalculateNet(int j)
        {
            double net = 0;
            for (int i = 1; i < weights.Length; ++i)
            {
                net += (weights[i] * fi[i]);
            }
            return net + weights[0];
        }

        /// <summary> Высчитывание вектора у ПОЛНОСТЬЮ </summary>
        private void CalculationALLFA()
        {
            for (int i = 0; i < t.Length; ++i)
            {
                Calculationfi(AllSets[i]);
                ThresholdFA(i);
            }
        }

        /// <summary> Высчитывание вектора у</summary>
        /// <param name="i">Позиция вектора у, которую нужно пересчитать</param>
        private void ThresholdFA(int i)
        {
            double net = СalculateNet(i);
            if (!flag)
                y[i] = (net >= 0) ? 1 : 0;

            else
            {
                double out_ = SigmoidalFunction(net);
                y[i] = (out_ >= 0.5) ? 1 : 0;
            }
        }

        /// <summary> Функция дифференциала для подсчета deltaWeights</summary>
        /// <param name="i">Индекс набора Х, для которого высчитывается net->дифференциал </param>
        private double Derivative(int i)
        {
            if (!flag)
                return 1;

            double net = СalculateNet(i);
            double Fnet = SigmoidalFunction(net);

            return Fnet * (1 - Fnet);

        }

        /// <summary> Подсчет значения сигмоидальной функции</summary>
        /// <param name="net">Значение net, для которого считается функция </param>
        private double SigmoidalFunction(double net)
        {
            net = net * -1.0;
            return 1.0 / (1.0 + Math.Exp(net));
        }

        /// <summary> Пересчитывания весов для каждого набора Х</summary>
        private void WidrowHoff()
        {
            for (int i = 0; i < sets_x.Length; ++i)
            {
                Calculationfi(sets_x[i]);
                ThresholdFA(i);
                for (int j = 0; j < weights.Length; ++j)
                {
                    deltaWeights[j] = teachingRate * (t[i] - y[i]) * fi[j] * Derivative(i);
                    weights[j] += deltaWeights[j];
                }
            }
        }

        private List<int>[] AllSets = { new List<int>{0,0,0,0}, new List<int>{0,0,0,1}, new List<int>{0,0,1,0}, new List<int>{0,0,1,1},
            new List<int>{0,1,0,0}, new List<int>{0,1,0,1}, new List<int>{0,1,1,0}, new List<int>{0,1,1,1}, new List<int>{1,0,0,0},
            new List<int>{1,0,0,1}, new List<int>{1,0,1,0}, new List<int>{1,0,1,1}, new List<int>{1,1,0,0}, new List<int>{1,1,0,1},
            new List<int>{1,1,1,0}, new List<int>{1,1,1,1}};

        private List<List<int>> C;
        private double teachingRate;
        private double[] t = new double[16];
        private double[] y = new double[16];
        private double[] weights;
        private double[] deltaWeights; 
        private double[] fi;
        private List<int>[] sets_x;
        private double Error;
        private int Epoch = 0;
        private bool flag; //Флаг для выбора Функции активации НС. False - пороговая ФА, true - сигмоидальная ФА
    }

    class Program
    {

        //Чтобы окно консоли не закрывалось запускать через Ctrl+F5
        static void Main(string[] args)
        {
            Console.WriteLine("Пороговая функция активации");
            RBF my = new RBF(0.3, new List<int>[] { new List<int> {0,0,0,0 },
            new List<int> {0,0,0,1 }, new List<int> {1,0,0,0 }, new List<int> { 1,0,1,0}, new List<int> {1,0,1,1},
            new List<int> {1,1,0,1 }}, false);
            my.Life();

            Console.WriteLine("__________________________________________________________" +
                              "__________________________________________________________" +
                "\n\nСигмоидальная функция активации\n");
            my = new RBF(0.3, new List<int>[] { new List<int> {0,0,0,0 },
            new List<int> {0,0,0,1 }, new List<int> {1,0,0,0 }, new List<int> { 1,0,1,0}, new List<int> {1,0,1,1},
            new List<int> {1,1,0,1 }}, true);
            my.Life();
        }
    }
}
