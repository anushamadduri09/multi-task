'''
This from https://github.com/dpressel/rude-carnie.
But changed by Anusha Madduri for masters project
to train the task-specidic and multi-task classfiers
in IMFDB dataset
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import math
import time
from data_multitasking import inputs
import numpy as np
import tensorflow as tf
from model_multitasking import select_model, get_checkpoint
import os
import json

tf.app.flags.DEFINE_string('train_dir', '',
                           'Training directory (where training data lives)')

tf.app.flags.DEFINE_integer('run_id', 0,
                           'This is the run number (pid) for training proc')

tf.app.flags.DEFINE_string('device_id', '/cpu:0',
                           'What processing unit to execute inference on')

tf.app.flags.DEFINE_string('eval_dir', '',
                           'Directory to put output to')

tf.app.flags.DEFINE_string('eval_data', 'valid',
                           'Data type (valid|train)')

tf.app.flags.DEFINE_integer('num_preprocess_threads', 4,
                            'Number of preprocessing threads')

tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', False,
                         """Whether to run eval only once.""")

tf.app.flags.DEFINE_integer('image_size', 227,
                            'Image size')

tf.app.flags.DEFINE_integer('batch_size', 128,
                            'Batch size')

tf.app.flags.DEFINE_integer('mloss', 0,
                            'chooses the type of loss function')

tf.app.flags.DEFINE_string('checkpoint', 'checkpoint',
                          'Checkpoint basename')


tf.app.flags.DEFINE_string('model_type', 'default',
                           'Type of convnet')

tf.app.flags.DEFINE_string('requested_step_seq', '', 'Requested step to restore')
FLAGS = tf.app.flags.FLAGS

        

def eval_once(saver, summary_writer, summary_op, logits, labels, num_eval, requested_step=None, mloss=0):
    """Run Eval once.
    Args:
    saver: Saver.
    summary_writer: Summary writer.
    top_k_op: Top K op.
    summary_op: Summary op.
    """

    age_labels, gender_labels, emotion_labels = labels
    logits_age, logits_gender, logits_emotion = logits

    # top1  and 2 for age
    top1_age = tf.nn.in_top_k(logits_age, age_labels, 1)
    top2_age = tf.nn.in_top_k(logits_age, age_labels, 2)

    # top1 and 2 for gender
    top1_gender = tf.nn.in_top_k(logits_gender, gender_labels, 1)
    top2_gender = tf.nn.in_top_k(logits_gender, gender_labels, 2)

    #top 1 and 2 for emotion
    top1_emotion = tf.nn.in_top_k(logits_emotion, emotion_labels, 1)
    top2_emotion = tf.nn.in_top_k(logits_emotion, emotion_labels, 2)

    with tf.Session() as sess:
        checkpoint_path = '%s/run-%d' % (FLAGS.train_dir, FLAGS.run_id)

        model_checkpoint_path, global_step = get_checkpoint(checkpoint_path, requested_step, FLAGS.checkpoint)

        #loads the model
        saver.restore(sess, model_checkpoint_path)

        # Start the queue runners.
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            num_steps = int(math.ceil(num_eval / FLAGS.batch_size))
            true_age_count1 = true_gender_count1 = true_emotion_count1 = 0
            total_sample_count = num_steps * FLAGS.batch_size
            step = 0
            print(FLAGS.batch_size, num_steps)

            while step < num_steps and not coord.should_stop():
                start_time = time.time()
                #runs the session and makes the predictions
                _, _, _, predictions1_age, predictions1_gender, predictions1_emotion = sess.run([logits_age, logits_gender, logits_emotion, top1_age, top1_gender, top1_emotion])
                duration = time.time() - start_time
                sec_per_batch = float(duration)
                examples_per_sec = FLAGS.batch_size / sec_per_batch
                
                #computed the total number of correctly predictions
                true_age_count1 += np.sum(predictions1_age)
                true_gender_count1 += np.sum(predictions1_gender)
                true_emotion_count1 += np.sum(predictions1_emotion)

                format_str = ('%s (%.1f examples/sec; %.3f sec/batch)')
                print(format_str % (datetime.now(),
                                    examples_per_sec, sec_per_batch))

                step += 1

            #computed the accuracy
            if(mloss == 0):
                age_accuracy = true_age_count1 / total_sample_count
                gender_accuracy = true_gender_count1 / total_sample_count
                emotion_accuracy = true_emotion_count1 / total_sample_count
                print('%s: age accuracy @ 1 = %.3f (%d/%d)' % (datetime.now(), age_accuracy, true_age_count1, total_sample_count))
                print('%s: gender accuracy @ 1 = %.3f (%d/%d)' % (datetime.now(), gender_accuracy, true_gender_count1, total_sample_count))
                print('%s: emotion accuracy @ 1 = %.3f (%d/%d)' % (datetime.now(), emotion_accuracy, true_emotion_count1, total_sample_count))
            elif(mloss == 1):
                age_accuracy = true_age_count1 / total_sample_count
                gender_accuracy = true_gender_count1 / total_sample_count
                print('%s: age accuracy @ 1 = %.3f (%d/%d)' % (datetime.now(), age_accuracy, true_age_count1, total_sample_count))
                print('%s: gender accuracy @ 1 = %.3f (%d/%d)' % (datetime.now(), gender_accuracy, true_gender_count1, total_sample_count))
            elif(mloss == 2):
                age_accuracy = true_age_count1 / total_sample_count
                emotion_accuracy = true_emotion_count1 / total_sample_count
                print('%s: age accuracy @ 1 = %.3f (%d/%d)' % (datetime.now(), age_accuracy, true_age_count1, total_sample_count))
                print('%s: emotion accuracy @ 1 = %.3f (%d/%d)' % (datetime.now(), emotion_accuracy, true_emotion_count1, total_sample_count))
            elif(mloss == 3):
                gender_accuracy = true_gender_count1 / total_sample_count
                emotion_accuracy = true_emotion_count1 / total_sample_count
                print('%s: gender accuracy @ 1 = %.3f (%d/%d)' % (datetime.now(), gender_accuracy, true_gender_count1, total_sample_count))
                print('%s: emotion accuracy @ 1 = %.3f (%d/%d)' % (datetime.now(), emotion_accuracy, true_emotion_count1, total_sample_count))


            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
        except Exception as e:  # pylint: disable=broad-except
            coord.request_stop(e)

        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate(run_dir):
    with tf.Graph().as_default() as g:
        input_file = os.path.join(FLAGS.train_dir, 'md.json')
        with open(input_file, 'r') as f:
            md = json.load(f)

        eval_data = FLAGS.eval_data == 'valid'
        num_eval = md['%s_counts' % FLAGS.eval_data]

        model_fn = select_model(FLAGS.model_type)


        with tf.device(FLAGS.device_id):
            print('Executing on %s' % FLAGS.device_id)
            images, labels_age, labels_gender, labels_emotion, _ = inputs(FLAGS.train_dir, FLAGS.batch_size, FLAGS.image_size, train=not eval_data, num_preprocess_threads=FLAGS.num_preprocess_threads)
            logits_age, logits_gender, logits_emotion = model_fn(md['nlabels_age'], md['nlabels_gender'], md['nlabels_emotion'], images, 1, False)
            summary_op = tf.summary.merge_all()

            logits = (logits_age, logits_gender, logits_emotion)
            labels = (labels_age, labels_gender, labels_emotion)
            
            summary_writer = tf.summary.FileWriter(run_dir, g)
            saver = tf.train.Saver()
            
            if FLAGS.requested_step_seq:
                sequence = FLAGS.requested_step_seq.split(',')
                for requested_step in sequence:
                    print('Running %s' % sequence)
                    eval_once(saver, summary_writer, summary_op, logits, labels, num_eval, requested_step)
            else:
                #while True:
                print('Running loop')
                eval_once(saver, summary_writer, summary_op, logits, labels, num_eval)
                #if FLAGS.run_once:
                #    break
                time.sleep(FLAGS.eval_interval_secs)

                
def main(argv=None):  # pylint: disable=unused-argument
    run_dir = '%s/run-%d' % (FLAGS.eval_dir, FLAGS.run_id)
    if tf.gfile.Exists(run_dir):
        tf.gfile.DeleteRecursively(run_dir)
    tf.gfile.MakeDirs(run_dir)
    evaluate(run_dir)


if __name__ == '__main__':
  tf.app.run()
