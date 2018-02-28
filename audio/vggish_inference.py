import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import tensorflow as tf

pca_params = './model/vggish_pca_params.npz'
checkpoint = './model/vggish_model.ckpt'

# Prepare a postprocessor to munge the model embeddings.
pproc = vggish_postprocess.Postprocessor(pca_params)
# Define the model in inference mode, load the checkpoint, and
# locate input and output tensors.
with tf.Graph().as_default() as g:
    sess = tf.Session(graph=g)
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

def audio_inference(wav_file):
    examples_batch = vggish_input.wavfile_to_examples(wav_file)
    with tf.Graph().as_default() as g:
        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})

        postprocessed_batch = pproc.postprocess(embedding_batch)

    return postprocessed_batch

# postprocessed_batch = audio_inference('../data/audio/00059.wav')