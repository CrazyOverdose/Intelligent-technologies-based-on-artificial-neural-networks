using System;
using System.Collections.Generic;

namespace ITIB1
{
    class NeuralNetwork
    {
        /// <summary> Конструктор класса Нейронной сети</summary>
        /// <param name="_flag">Флаг для выбора Функции активации НС.
        /// False - пороговая ФА, true - сигмоидальная ФА</param>
        /// <param name="_teachingRate">Норма обучения</param>
        /// <param name="_weights">Начальные весовые коэффициенты</param>
        /// <param name="_t">Целевой набор</param>
        public NeuralNetwork(double _teachingRate, bool _flag, double[] _weights, double[] _t, List<int>[] sets_)
        {
            teachingRate = _teachingRate;
            flag = _flag;
            weights = _weights;
            t = _t;
            sets = sets_;

            CalculationALLFA();
            TotalError();
        }

        ~NeuralNetwork() { }

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

        /// <summary> Поиск лучшего набора х</summary>
        /// <param name="MinEpoch">Минимальное количество эпох</param>
        /// <returns>true - найден лучший набор х, false - не найден</returns>
        public bool Search(int MinEpoch)
        {
            while (true)
            {
                if (Epoch >= MinEpoch)
                    return false;

                if (Error == 0)
                    return true;

                WidrowHoff();
                ++Epoch;
                CalculationALLFA();
                TotalError();
            }
        }

        /// <summary> Высчитывание net</summary>
        /// <param name="j">Индекс  набора Х, для которого нужно подсчитать net</param>
        private double СalculateNet(int j)
        {
            double net = 0;
            for (int i = 0; i < 5; ++i)
            {
                net += (weights[i] * AllSets[j][i]);
            }
            return net;
        }

        // <summary> Высчитывание вектора у ПОЛНОСТЬЮ</summary>
        private void CalculationALLFA()
        {
            for (int i = 0; i < 16; ++i)
            {
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

        /// <summary> Высчитывание тотальной ошибки</summary>
        private void TotalError()
        {
            int error = 0;
            for (int i = 0; i < 16; ++i)
            {
                error += (y[i] == t[i]) ? 0 : 1;
            }

            Error = error;
        }

        /// <summary> Пересчитывания весов для каждого набора Х</summary>
        private void WidrowHoff()
        {
            for (int i = 0; i < sets.Length; ++i)
            {
                ThresholdFA(i);
                for (int j = 0; j < weights.Length; ++j)
                {

                    deltaWeights[j] = teachingRate * (t[i] - y[i]) * sets[i][j] * Derivative(i);
                    weights[j] += deltaWeights[j];
                }
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

        public List<int>[] GetSets()
        {
            return sets;
        }

        public int GetEpoch()
        {
            return Epoch;
        }

        private double teachingRate; //Норма обучения
        private double[] weights = new double[5];
        private double[] deltaWeights = new double[5];
        private double[] t = new double[16]; //Целевой набор
        private double[] y = new double[16];
        private List<int>[] AllSets = { new List<int>{1,0,0,0,0}, new List<int>{1,0,0,0,1}, new List<int>{1,0,0,1,0}, new List<int>{1,0,0,1,1},
            new List<int>{1,0,1,0,0}, new List<int>{1,0,1,0,1}, new List<int>{1,0,1,1,0}, new List<int>{1,0,1,1,1}, new List<int>{1,1,0,0,0},
            new List<int>{1,1,0,0,1}, new List<int>{1,1,0,1,0}, new List<int>{1,1,0,1,1}, new List<int>{1,1,1,0,0}, new List<int>{1,1,1,0,1},
            new List<int>{1,1,1,1,0}, new List<int>{1,1,1,1,1}};

        private List<int>[] sets;
        private bool flag; //Флаг для выбора Функции активации НС. False - пороговая ФА, true - сигмоидальная ФА
        private int Error = -1; //Суммарная квадратичная ошибка
        private int Epoch = 0; //Эпоха
    }

    class Program
    {
        static List<int>[] SearchForAllCombinations(int MinEpoch, List<int>[] AllSets, int[] ArrayIndex, bool flag)
        {
            List<int>[] BestSets = null; // лучший набор х

            for (int i = 0; i < 17; ++i)
            {
                Сombinations S = new Сombinations();

                var AllCombinations = S.AllCombinations(ArrayIndex, 15, i); // все комбинации индексов из i индексов

                for (int j = 0; j < AllCombinations.Count; ++j)
                {
                    List<int>[] Combination = new List<int>[AllCombinations[j].Length]; // заполнение набора х по индексам

                    for (int k = 0; k < AllCombinations[j].Length; ++k)
                    {
                        Combination[k] = AllSets[AllCombinations[j][k]];
                    }

                    NeuralNetwork my = new NeuralNetwork(0.3, flag, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 },
                    new double[] { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0 }, Combination);

                    if (my.Search(MinEpoch))
                    {
                        MinEpoch = my.GetEpoch();
                        BestSets = my.GetSets();
                    }
                }
            }

            return BestSets;
        }

        //Чтобы окно консоли не закрывалось запускать через Ctrl+F5
        static void Main(string[] args)
        {
            List<int>[] AllSets = { new List<int>{1,0,0,0,0}, new List<int>{1,0,0,0,1}, new List<int>{1,0,0,1,0}, new List<int>{1,0,0,1,1},
                    new List<int>{1,0,1,0,0}, new List<int>{1,0,1,0,1}, new List<int>{1,0,1,1,0}, new List<int>{1,0,1,1,1}, new List<int>{1,1,0,0,0},
                    new List<int>{1,1,0,0,1}, new List<int>{1,1,0,1,0}, new List<int>{1,1,0,1,1}, new List<int>{1,1,1,0,0}, new List<int>{1,1,1,0,1},
                    new List<int>{1,1,1,1,0}, new List<int>{1,1,1,1,1}};

            Console.WriteLine("Пороговая функция активации");
            NeuralNetwork my = new NeuralNetwork(0.3, false, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 },
                new double[] { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0 }, AllSets);
            my.Life();

            Console.WriteLine("__________________________________________________________" +
                              "__________________________________________________________" +
                "\n\nСигмоидальная функция активации\n");
            my = new NeuralNetwork(0.3, true, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 },
                new double[] { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0 }, AllSets);
            my.Life();

            int[] ArrayIndex = { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 }; // массив индексов наборов х

            Console.WriteLine("__________________________________________________________" +
                              "__________________________________________________________" +
                "\n\nПороговая функция активации при лучшей комбинации х\n");
            List<int>[] BestSets = SearchForAllCombinations(20, AllSets, ArrayIndex, false);
            my = new NeuralNetwork(0.3, false, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 },
                new double[] { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0 }, BestSets);
            my.Life();

            Console.WriteLine("__________________________________________________________" +
                              "__________________________________________________________" +
                "\n\nСигмоидальная функция активации при лучшей комбинации х\n");
            BestSets = SearchForAllCombinations(20, AllSets, ArrayIndex, true);
            my = new NeuralNetwork(0.3, true, new double[] { 0.0, 0.0, 0.0, 0.0, 0.0 },
                new double[] { 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0 }, BestSets);
            my.Life();
        }
    }

    class Сombinations

    {
        private List<int[]> AllCombinations_ = new List<int[]>();

        private void Сombination(int[] ArrayIndex, int[] IntermediateArray, int Start, int End, int Index, int m)
        {

            if (Index == m)

            {
                int[] Combination = new int[m];
                for (int j = 0; j < m; j++)

                    Combination[j] = IntermediateArray[j];

                AllCombinations_.Add(Combination);

                return;
            }

            for (int i = Start; i <= End && End - i + 1 >= m - Index; i++)

            {
                IntermediateArray[Index] = ArrayIndex[i];

                Сombination(ArrayIndex, IntermediateArray, i + 1, End, Index + 1, m);
            }

        }

        public List<int[]> AllCombinations(int[] ArrayIndex, int n, int m)

        {
            int[] IntermediateArray = new int[m];

            Сombination(ArrayIndex, IntermediateArray, 0, n - 1, 0, m);

            return AllCombinations_;
        }
    }

}
