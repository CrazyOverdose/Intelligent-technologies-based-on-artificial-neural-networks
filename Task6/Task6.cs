using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml;

namespace Нейронки_9
{
    struct Theatre
    {
        public string Name;
        public int Capacity;
    }


    class ClusterAnalysis
    {
        /// <summary>
        /// Конструктор класса кластерного анализа данных
        /// </summary>
        /// <param name="Link"> Путь, по которому лежит Xml с театрами г. Москвы</param>
        public ClusterAnalysis(string Link)
        {
            Encoding win1251 = Encoding.GetEncoding(1251);
            XmlDocument xDoc = new XmlDocument();
            xDoc.Load(Link);

            // получим корневой элемент
            XmlElement xRoot = xDoc.DocumentElement;

            // обход всех узлов в корневом элементе
            foreach (XmlNode xnode in xRoot)
            {
                Theatre new_theatre = new Theatre();

                // обходим все дочерние узлы элемента user
                foreach (XmlNode childnode in xnode.ChildNodes)
                {
                    // если узел - название театра
                    if (childnode.Name == "CommonName")
                    {
                        new_theatre.Name = childnode.InnerText.Replace("«", "\"").Replace("»", "\"");
                    }
                    // если узел - Вместимость главного зала
                    else if (childnode.Name == "MainHallCapacity")
                    {
                        new_theatre.Capacity += Convert.ToInt32(childnode.InnerText);
                    }

                    // если узел - Дополнительная вместимость зала
                    else if (childnode.Name == "AdditionalHallCapacity")
                    {
                        new_theatre.Capacity += Convert.ToInt32(childnode.InnerText);
                    }
                }
               // Удаляем из выборки  театры, где не указана вместимость залов
               if(new_theatre.Capacity != 0)
                   Theaters.Add(new_theatre);
            }

            ClusterCreation();
            NSKohonen();
            PrintResult();
        }

        /// <summary>
        /// Нейросеть Кохонена
        /// </summary>
        private void NSKohonen()
        {
            foreach (var i in  Theaters)
            {
                var dist = 2147483647; // максимальное возможное расстояние
                var index = 0; // Номер кластера, к которому принадлежит данный театр

                for (var j = 0; j < Clusters.Count; ++j)
                {
                    var Distance = DistanceSquared(i, Clusters[j]);

                    if(Distance < dist)
                    {
                        dist = Distance;
                        index = j;
                    }
                }

                ClusterMembership[index].Add(i);
            }
        }

        /// <summary>
        /// Расчёт кластеров
        /// </summary>
        private void ClusterCreation()
        {
            var MaxCapacity = Theaters.Max(Theater_ => Theater_.Capacity);
            var MinCapacity = Theaters.Min(Theater_ => Theater_.Capacity);
            var Difference = Math.Ceiling(((MaxCapacity - MinCapacity) / 10.0) / 10) * 10;

            for (var i = Difference; i <= MaxCapacity; i += Difference)
            {
                Clusters.Add(Convert.ToInt32(i));
            }

            foreach (var i in  Clusters)
                ClusterMembership.Add(new List<Theatre>());
        }

        /// <summary>
        /// Расчет расстояния
        /// </summary>
        /// <param name="x">Вместимость театра</param>
        /// <param name="w">Кластер</param>
        /// <returns></returns>
        private int DistanceSquared(Theatre x, int w)
        {
            return w * w - 2 * x.Capacity * w;
        }

        /// <summary>
        /// Печать результата
        /// </summary>
        private void PrintResult()
        {
            for(var i = 0; i < Clusters.Count; ++i)
            {
                Console.Write($"Кластер {i+1} - ");
                Console.WriteLine($"вместимость залов театра равна {Clusters[i]}");
                Console.WriteLine("________________________________________________\n");
                Console.WriteLine("К данному кластеру принадлежат:");
                foreach(var j in ClusterMembership[i])
                {
                    Console.WriteLine($"\nНазвание театра: {j.Name}");
                    Console.WriteLine($"Вместимость театра: {j.Capacity}");
                }
                Console.WriteLine();
            }
        }

        ~ClusterAnalysis() { }

        private List<Theatre> Theaters = new List<Theatre>(); // Список всех театров
        private List<int> Clusters = new List<int>(); // Кластеры
        private List<List<Theatre>> ClusterMembership = new List<List<Theatre>>(); // Принадлежность театров к кластерам

    }
    class Program
    {
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Black;
            Console.BackgroundColor = ConsoleColor.White;
            ClusterAnalysis my = new ClusterAnalysis("./list.xml");
        }

    }
}
