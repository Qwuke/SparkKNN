import org.apache.spark.api.java.*;
import org.apache.spark.SparkConf;
import java.util.*;
import scala.Tuple2;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import static java.util.Map.Entry;


public class SparkKNN {
    public static void main(String[] args) {
        SparkConf sparkConf = new SparkConf().setAppName("KNN Spark");
        JavaSparkContext sc = new JavaSparkContext(sparkConf);
        if (args.length < 4) {
            System.err.println(
                    "use: knnspark <k> <path: Labeled Data.txt> <path: Raw Data.txt> <hdfs://ip:port/path/final_filename.txt>");
            System.exit(1);
        }
        int numNN = Integer.parseInt((args[0]));
        String trainData = args[1];
        String rawData = args[2];
        JavaRDD<Double[]> trainPoints = sc.textFile(trainData).map(l -> l.split(" ")).map(x -> {
            int i = 0;
            Double[] trainPoint = new Double[x.length];
            for (String a : x) {
                trainPoint[i] = Double.parseDouble(a);
                i++;
            }
            return trainPoint;
        });
        JavaRDD<Double[]> rawPoints = sc.textFile(rawData).map(l -> l.split(" ")).map(x -> {
            int i = 0;
            Double[] rawPoint = new Double[x.length];
            for (String a : x) {
                rawPoint[i] = Double.parseDouble(a);
                i++;
            }
            return rawPoint;
        });
        JavaRDD<Double[]> newPoints = rawPoints.map(z -> {
            int len = z.length;
            JavaPairRDD<Double, String> distances = trainPoints.mapToPair(y -> {
                Double sum = 0.0;
                for (int i = 0; i < len - 1; i++) {
                    sum = sum + Math.pow((y[i] - z[i]), 2);
                }
                String label = y[y.length - 1].toString();
                return new Tuple2<>(Math.sqrt(sum), label);
            });
            List<Tuple2<Double, String>> sortDist = distances.sortByKey(true).take(numNN);
            Map<String, Integer> labelCount = new HashMap<>();
            for (Tuple2<Double, String> k: sortDist) {
                Integer n = labelCount.get(k._2());
                n = (n == null) ? 1 : ++n;
                labelCount.put(k._2(), n);
            }
            int maxValueInMap=(Collections.max(labelCount.values()));
            ArrayList<String> voteLabels = new ArrayList<>();
            for (Entry<String, Integer> entry : labelCount.entrySet()) {
                if (entry.getValue()==maxValueInMap) {
                     voteLabels.add(entry.getKey());
                }}
            String voteLabel;
            if(voteLabels.size() == 1) {
                voteLabel = voteLabels.get(0);
            } else {
              voteLabel = voteLabels.get((int )(Math.random() * voteLabels.size()));
            }
            Double[] newPoint = new Double[len+1];
            System.arraycopy(z, 0, newPoint, 0, len+1);
            newPoint[len] = Double.parseDouble(voteLabel);
            return newPoint;
            });
        System.out.println("Points have been classified, writing new text file to: " + args[3] );
        newPoints.saveAsTextFile(args[3]);
        sc.stop();
        }


}