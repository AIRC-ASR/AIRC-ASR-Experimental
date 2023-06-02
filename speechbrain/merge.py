import difflib
from nltk.metrics.distance import edit_distance

def mergeHypotheses(reference, hypothesis1, hypothesis2):
    merged = ""
    reference_words = reference.split()
    hypothesis1_words = hypothesis1.split()
    hypothesis2_words = hypothesis2.split()
    errors = 0

    for i in range(len(reference_words)):
        if i < len(hypothesis1_words):
            if hypothesis1_words[i] != reference_words[i]:
                if i < len(hypothesis2_words):
                    if hypothesis2_words[i] != reference_words[i]:
                        merged += hypothesis2_words[i] + " "
                        errors += 1
                    else:
                        merged += hypothesis1_words[i] + " "
                else:
                    merged += hypothesis1_words[i] + " "
                    errors += 1
            else:
                merged += hypothesis1_words[i] + " "
        elif i < len(hypothesis2_words):
            if hypothesis2_words[i] != reference_words[i]:
                merged += hypothesis2_words[i] + " "
                errors += 1
            else:
                merged += reference_words[i] + " "
        else:
            merged += reference_words[i] + " "

    merged_hypothesis = merged.strip()
    hypothesis1_wer = calculateWER(reference, hypothesis1)
    merged_wer = calculateWER(reference, merged_hypothesis)

    if merged_wer > hypothesis1_wer:
        return hypothesis1
    else:
        return merged_hypothesis


def calculateWER(reference, hypothesis):
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    errors = 0

    for i in range(len(reference_words)):
        if i < len(hypothesis_words):
            if hypothesis_words[i] != reference_words[i]:
                errors += 1
        else:
            errors += 1

    return errors / len(reference_words)




def calculateWER(reference, hypothesis):
    reference_words = reference.split()
    hypothesis_words = hypothesis.split()
    num_reference_words = len(reference_words)

    # Calculate the edit distance between the reference and hypothesis sentences
    edit_dist = edit_distance(reference_words, hypothesis_words)

    # Calculate the WER as the edit distance divided by the number of reference words
    wer = edit_dist / num_reference_words * 100

    return wer

if __name__ == '__main__':
  # Test Case A
  reference_a = "CUT AS MANY NICE EVEN SLICES AS MAY BE REQUIRED RATHER MORE THAN ONE QUARTER INCH IN THICKNESS AND TOAST THEM BEFORE A VERY BRIGHT FIRE WITHOUT ALLOWING THE BREAD TO BLACKEN WHICH SPOILS THE APPEARANCE AND FLAVOUR OF ALL TOAST"
  hypothesis1_a = "CUT AS MANY NICE EVEN SLICES AS MAY BE REQUIRED RATHER MORE THAN ONE QUARTER INCH IN THICKNESS AND TOAST THEM BEFORE A VERY BRIGHT FIRE WITHOUT ALLOWING THE BREAD TO BLACKEN WHICH SPOILS THE APPEARANCE AND FLAVOR OF ALL TOAST"
  hypothesis2_a = "CUT AS MANY NICE  EVEN SLICES ASMAY BE REQQUIRED RATHERMORE THAN ONE QUARTER INCH IN THICKNESSAND TOAST THEMBEFORE A VERYBRIGHT FIRE WITHOUT ALLOWING THE BREADTO  BLACKEN WHICHSPOILS THEAPPEARANCEAND  FLAVOUROFALL TOAST"

  # Test Case B
  reference_b = "BUT THE RASHNESS OF THESE CONCESSIONS HAS ENCOURAGED A MILDER SENTIMENT OF THOSE OF THE DOCETES WHO TAUGHT NOT THAT CHRIST WAS A PHANTOM BUT THAT HE WAS CLOTHED WITH AN IMPASSIBLE AND INCORRUPTIBLE BODY"
  hypothesis1_b = "BUT THE RASHNESS OF THESE CONCESSIONS HAS ENCOURAGED A MILDER SENTIMENT OF THOSE OF THE DOCITS WHO TAUGHT NOT THAT CHRIST WAS A PHANTOM BUT THAT HE WAS CLOTHED WITH AN IMPASSIBLE AND INCORRUPTIBLE BODY"
  hypothesis2_b = "BUT THE RASHNESS OFTHESE CONCESSIONS HASENCOURAGEDA MILDER SENTIMENT OF THOSE OF THE DOSETESWHO TAUGHT NOT THATCHRIST WAS A PHANTOM BUTTHATHE WAS  CLOTHED WITH AN IMPASSABLE AND INCORRUPTIBLE BODY"

  # Test Case C
  reference_c = "I FREQUENTLY THOUGHT IT WOULD BE PLEASANT TO SPLIT THE DIFFERENCE WITH THAT MULE AND I WOULD GLADLY HAVE DONE SO IF I COULD HAVE GOTTEN ONE HALF OF HIS NO"
  hypothesis1_c = "I FREQUENTLY THOUGHT IT WOULD BE PLEASANT TO SPLIT THE DIFFERENCE WITH THAT MULE AND I WOULD GLADLY HAVE DONE SO IF I COULD HAVE GOTTEN ONE HALF OF HIS NOW"
  hypothesis2_c = "I FREQUENTLY THOUGHT IT WOULDBEPLEASANT TO SPLIT THE DIFFERENCE WITH THAT MULE AND I WOULDGLADLYHAVE DONE SOIFI COULDHAVE GOTTENONE HALFOF HIS KNOE"

  # Test Case D
  reference_d = "O WISE MOTHER MAGPIE DEAR MOTHER MAGPIE THEY CRIED TEACH US HOW TO BUILD OUR NESTS LIKE YOURS FOR IT IS GROWING NIGHT AND WE ARE TIRED AND SLEEPY"
  hypothesis1_d = "OH WISE MOTHER MAGPIE DEAR MOTHER MAGPIE THEY CRIED TEACH US HOW TO BUILD OUR NESTS LIKE YOURS FOR IT IS GROWING NIGHT AND WE ARE TIRED AND SLEEPY"
  hypothesis2_d = "OH WISE MOTHER MAGPIE DEAR MOTHER MAGPIETHEY CRIED TEACHUS HOW TO BUILD OUR NESTS LIKEYOURS FOR IT IS  GROWING NIGHTAND  WE ARETIRED AND SLEEPY"

  # Test Case E
  reference_e = "AND THEREWITHAL SUCH A BEWILDERMENT POSSESS'D ME THAT I SHUT MINE EYES FOR PEACE AND IN MY BRAIN DID CEASE ORDER OF THOUGHT AND EVERY HEALTHFUL THING"
  hypothesis1_e = "AND THEREWITHAL SUCH A BEWILDERMENT POSSESSED ME THAT I SHUT MINE EYES FOR PEACE AND IN MY BRAIN DID CEASE ORDER OF THOUGHT AND EVERY HEALTHFUL THING"
  hypothesis2_e = "ANDTHEREWITHAL SUCH A BEWILDERMENTPOSSESSED ME THATI SHUT MINE EYES FOR PEACEAND IN MY BRAINDID CEASEORDER OF THOUGHTANDEVERY HEALTHFUL THING"

  # Test Case F
  reference_f = "A HARSH LAUGH FROM COMRADE OSSIPON CUT THE TIRADE DEAD SHORT IN A SUDDEN FALTERING OF THE TONGUE AND A BEWILDERED UNSTEADINESS OF THE APOSTLE'S MILDLY EXALTED EYES"
  hypothesis1_f = "A HARSH LAUGH FROM CONRAD OSSIPAN CUT THE TIRADE DEAD SHORT IN A SUDDEN FALTERING OF THE TONGUE AND THE BEWILDERED UNSTEADINESS OF THE APOSTLE'S MILDLY EXALTED EYES"
  hypothesis2_f = "A HARSH LAUGH FROM COMRADE ASCEPEN CUT THE TYRADE DEAD SHORT IN A SUDDENFALTERING OF THE TONGUE IN THAE BEWILDEREDUNSTEADINESSOF THE APOSTLE'SMILDLY EXALTED EYES"

  # Test Case G
  reference_g = "I BELIEVE SAID JOHN THAT IN THE SIGHT OF GOD I HAVE A RIGHT TO FULFILL THAT PROMISE"
  hypothesis1_g = "I BELIEVE SAID JOHN THAT IN THE SIGHT OF GOD I HAVE A RIGHT TO FULFIL THAT PROMISE"
  hypothesis2_g = "IBELIEVE SAID  JOHN THATIN THE SIGHT OF GODI HAVE A RIGHT TO FULFIL THAT PROMISE"

  # Test Case H
  reference_h = "TRACING THE MC CLOUD TO ITS HIGHEST SPRINGS AND OVER THE DIVIDE TO THE FOUNTAINS OF FALL RIVER NEAR FORT CROOK THENCE DOWN THAT RIVER TO ITS CONFLUENCE WITH THE PITT ON FROM THERE TO THE VOLCANIC REGION ABOUT LASSEN'S BUTTE THROUGH THE BIG MEADOWS AMONG THE SOURCES OF THE FEATHER RIVER AND DOWN THROUGH FORESTS OF SUGAR PINE TO THE FERTILE PLAINS OF CHICO THIS IS A GLORIOUS SAUNTER AND IMPOSES NO HARDSHIP"
  hypothesis1_h = "TRACING THE MC CLOUD TO ITS HIGHEST SPRINGS AND OVER THE DIVIDE TO THE FOUNTAINS OF FALL RIVER NEAR FORT CROOK THENCE DOWN THAT RIVER TO ITS CONFLUENCE WITH THE PIT ON FROM THERE TO THE VOLCANIC REGION ABOUT LASSON'S BUTTE THROUGH THE BIG MEADOWS AMONG THE SOURCES OF THE FEATHER RIVER AND DOWN THROUGH FORESTS OF SUGAR PINE TO THE FERTILE PLAINS OF CHICO THIS IS A GLORIOUS SAUNTER AND IMPOSES NO HARDSHIP"
  hypothesis2_h = "TRACING THEMCCLOUDTOITS HIGHEST SPRINGS AND OVER THE DIVIDETO THE FOUNTAINS OF FALL RIVERNEAR FORT CROOKE THENCE DOWN THAT  RIVER TO ITS CONFLUENCEWITH THE PITTON FROMTHERE TO THE VOLCANIC REGIONABOUT LASCENS BUTTETHROUGH THE BIG MEADOWSAMONGTHE SOURCES OF THE FEATHER RIVERAND DOWN THROUGH  FORESTSOF  SUGAR PINETOTHE FERTILEPLAINS OF CHEEKO THIS IS A GLORIOUS PAUNTER ANDIMPOSES  NO HARDSHIP"

  # Test Case I
  reference_i = "THE LEADEN HAIL STORM SWEPT THEM OFF THE FIELD THEY FELL BACK AND RE FORMED"
  hypothesis1_i = "THE LEADEN HAILSTORM SWEPT THEM OFF THE FIELD THEY FELL BACK AND REFORMED"
  hypothesis2_i = "THE LEADEN HAIL STORM  SWEPTTHEM OFF THE FIELD THEYFELL BACKAND REFORMED"

  ##### Baseline WERs #####
  references = [reference_a, reference_b, reference_c, reference_d, reference_e, reference_f, reference_g, reference_h, reference_i]
  hypothesis1s = [hypothesis1_a, hypothesis1_b, hypothesis1_c, hypothesis1_d, hypothesis1_e, hypothesis1_f, hypothesis1_g, hypothesis1_h, hypothesis1_i]
  hypothesis2s = [hypothesis2_a, hypothesis2_b, hypothesis2_c, hypothesis2_d, hypothesis2_e, hypothesis2_f, hypothesis2_g, hypothesis2_h, hypothesis2_i]
  merged_hypotheses = [
    mergeHypotheses(reference_a, hypothesis1_a, hypothesis2_a),
    mergeHypotheses(reference_b, hypothesis1_b, hypothesis2_b),
    mergeHypotheses(reference_c, hypothesis1_c, hypothesis2_c),
    mergeHypotheses(reference_d, hypothesis1_d, hypothesis2_d),
    mergeHypotheses(reference_e, hypothesis1_e, hypothesis2_e),
    mergeHypotheses(reference_f, hypothesis1_f, hypothesis2_f),
    mergeHypotheses(reference_g, hypothesis1_g, hypothesis2_g),
    mergeHypotheses(reference_h, hypothesis1_h, hypothesis2_h),
    mergeHypotheses(reference_i, hypothesis1_i, hypothesis2_i)
  ]
  print("Baseline WERs:")
  for i in range(len(references)):
    print("Reference:", references[i])
    print("Hypothesis 1:", hypothesis1s[i])
    print("Hypothesis 2:", hypothesis2s[i])
    print("Merged Hypothesis:", merged_hypotheses[i])
    print("WER 1:", calculateWER(references[i], hypothesis1s[i]))
    print("WER 2:", calculateWER(references[i], hypothesis2s[i]))
    print("WER Merged:", calculateWER(references[i], merged_hypotheses[i]))
    print()

  ##### Find the merged hypothesis #####
  # # Test Case A
  # merged_a = 
  # print("Merged Hypothesis A:", merged_a)

  # # Test Case B
  # merged_b = mergeHypotheses(reference_b, hypothesis1_b, hypothesis2_b)
  # print("Merged Hypothesis B:", merged_b)

  # # Test Case C
  # merged_c = mergeHypotheses(reference_c, hypothesis1_c, hypothesis2_c)
  # print("Merged Hypothesis C:", merged_c)

  # # Test Case D
  # merged_d = mergeHypotheses(reference_d, hypothesis1_d, hypothesis2_d)
  # print("Merged Hypothesis D:", merged_d)

  # # Test Case E
  # merged_e = mergeHypotheses(reference_e, hypothesis1_e, hypothesis2_e)
  # print("Merged Hypothesis E:", merged_e)

  # # Test Case F
  # merged_f = mergeHypotheses(reference_f, hypothesis1_f, hypothesis2_f)
  # print("Merged Hypothesis F:", merged_f)

  # # Test Case G
  # merged_g = mergeHypotheses(reference_g, hypothesis1_g, hypothesis2_g)
  # print("Merged Hypothesis G:", merged_g)

  # # Test Case H
  # merged_h = mergeHypotheses(reference_h, hypothesis1_h, hypothesis2_h)
  # print("Merged Hypothesis H:", merged_h)

  # # Test Case I
  # merged_i = mergeHypotheses(reference_i, hypothesis1_i, hypothesis2_i)
  # print("Merged Hypothesis I:", merged_i)