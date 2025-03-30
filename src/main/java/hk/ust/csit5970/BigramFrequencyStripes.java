package hk.ust.csit5970;

import java.io.IOException;
import java.util.Arrays;
import java.util.Map;

import org.apache.commons.cli.CommandLine;
import org.apache.commons.cli.CommandLineParser;
import org.apache.commons.cli.GnuParser;
import org.apache.commons.cli.HelpFormatter;
import org.apache.commons.cli.OptionBuilder;
import org.apache.commons.cli.Options;
import org.apache.commons.cli.ParseException;
import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.conf.Configured;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.FloatWritable;
import org.apache.hadoop.io.IntWritable;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.Mapper;
import org.apache.hadoop.mapreduce.Reducer;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;
import org.apache.hadoop.util.Tool;
import org.apache.hadoop.util.ToolRunner;
import org.apache.log4j.Logger;

public class BigramFrequencyStripes extends Configured implements Tool {
    private static final Logger LOG = Logger.getLogger(BigramFrequencyStripes.class);

    private static class MyMapper extends Mapper<LongWritable, Text, Text, HashMapStringIntWritable> {
        private static final Text KEY = new Text();
        private static final HashMapStringIntWritable STRIPE = new HashMapStringIntWritable();

        @Override
        public void map(LongWritable key, Text value, Context context)
                throws IOException, InterruptedException {
            String line = value.toString();
            String[] words = line.trim().split("\\s+");
            for (int i = 0; i < words.length - 1; i++) {
                String w1 = words[i];
                String w2 = words[i + 1];
                KEY.set(w1);
                STRIPE.clear();
                STRIPE.put(w2, 1);
                context.write(KEY, STRIPE);
            }
        }
    }

    private static class MyReducer extends Reducer<Text, HashMapStringIntWritable, PairOfStrings, FloatWritable> {
        private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();
        private final static PairOfStrings BIGRAM = new PairOfStrings();
        private final static FloatWritable FREQ = new FloatWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context)
                throws IOException, InterruptedException {
            SUM_STRIPES.clear();
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                    SUM_STRIPES.increment(entry.getKey(), entry.getValue());
                }
            }

            int total = 0;
            for (int count : SUM_STRIPES.values()) {
                total += count;
            }

            // Emit total for A
            BIGRAM.set(key.toString(), "");
            FREQ.set(total);
            context.write(BIGRAM, FREQ);

            // Emit each B's frequency
            for (Map.Entry<String, Integer> entry : SUM_STRIPES.entrySet()) {
                String b = entry.getKey();
                int count = entry.getValue();
                float freq = (float) count / total;
                BIGRAM.set(key.toString(), b);
                FREQ.set(freq);
                context.write(BIGRAM, FREQ);
            }
        }
    }

    private static class MyCombiner extends Reducer<Text, HashMapStringIntWritable, Text, HashMapStringIntWritable> {
        private final static HashMapStringIntWritable SUM_STRIPES = new HashMapStringIntWritable();

        @Override
        public void reduce(Text key, Iterable<HashMapStringIntWritable> stripes, Context context)
                throws IOException, InterruptedException {
            SUM_STRIPES.clear();
            for (HashMapStringIntWritable stripe : stripes) {
                for (Map.Entry<String, Integer> entry : stripe.entrySet()) {
                    SUM_STRIPES.increment(entry.getKey(), entry.getValue());
                }
            }
            context.write(key, SUM_STRIPES);
        }
    }

    public BigramFrequencyStripes() {
    }

    private static final String INPUT = "input";
    private static final String OUTPUT = "output";
    private static final String NUM_REDUCERS = "numReducers";

    @Override
    public int run(String[] args) throws Exception {
        Options options = new Options();
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("input path").create(INPUT));
        options.addOption(OptionBuilder.withArgName("path").hasArg().withDescription("output path").create(OUTPUT));
        options.addOption(OptionBuilder.withArgName("num").hasArg().withDescription("number of reducers").create(NUM_REDUCERS));

        CommandLine cmdline;
        CommandLineParser parser = new GnuParser();
        try {
            cmdline = parser.parse(options, args);
        } catch (ParseException exp) {
            System.err.println("Error parsing command line: " + exp.getMessage());
            return -1;
        }

        if (!cmdline.hasOption(INPUT) || !cmdline.hasOption(OUTPUT)) {
            HelpFormatter formatter = new HelpFormatter();
            formatter.setWidth(120);
            formatter.printHelp(this.getClass().getName(), options);
            ToolRunner.printGenericCommandUsage(System.out);
            return -1;
        }

        String inputPath = cmdline.getOptionValue(INPUT);
        String outputPath = cmdline.getOptionValue(OUTPUT);
        int reduceTasks = cmdline.hasOption(NUM_REDUCERS) ? Integer.parseInt(cmdline.getOptionValue(NUM_REDUCERS)) : 1;

        Configuration conf = getConf();
        Job job = Job.getInstance(conf);
        job.setJobName(BigramFrequencyStripes.class.getSimpleName());
        job.setJarByClass(BigramFrequencyStripes.class);
        job.setNumReduceTasks(reduceTasks);

        FileInputFormat.setInputPaths(job, new Path(inputPath));
        FileOutputFormat.setOutputPath(job, new Path(outputPath));

        job.setMapOutputKeyClass(Text.class);
        job.setMapOutputValueClass(HashMapStringIntWritable.class);
        job.setOutputKeyClass(PairOfStrings.class);
        job.setOutputValueClass(FloatWritable.class);

        job.setMapperClass(MyMapper.class);
        job.setCombinerClass(MyCombiner.class);
        job.setReducerClass(MyReducer.class);

        FileSystem.get(conf).delete(new Path(outputPath), true);

        long startTime = System.currentTimeMillis();
        job.waitForCompletion(true);
        LOG.info("Job Finished in " + (System.currentTimeMillis() - startTime) / 1000.0 + " seconds");

        return 0;
    }

    public static void main(String[] args) throws Exception {
        ToolRunner.run(new BigramFrequencyStripes(), args);
    }
}

