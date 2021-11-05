# DialFact: A Benchmark for Fact-Checking in Dialogue
Authors: [Prakhar Gupta](https://prakharguptaz.github.io/), [Jason Wu](https://jasonwu0731.github.io/), Wenhao Liu and Caiming Xiong

Paper link: https://arxiv.org/pdf/2110.08222


## Abstract
To study the problem of Fact-Checking in Dialogue, we construct and introduce DIALFACT, a testing benchmark dataset of 22,123 annotated conversational claims, paired with pieces of evidence from Wikipedia. There are three sub-tasks in DIALFACT: 1) Verifiable claim detection task distinguishes whether a response carries verifiable factual information; 2) Evidence retrieval task retrieves the most relevant Wikipedia snippets as evidence; 3) Claim verification task predicts a dialogue response to be supported, refuted, or not enough information.



------------

## Dataset Details

The statistics for the Test and Validation sets are shown in the figure below. 
(Note that the numbers are a bit different from the first version of arxiv draft). 
![Data stats](/images/stats.png?raw=true "Data Stats")


### Data format
Description of keys and values present in the dataset files:
```json
{
    "context_id": "Context ID",
    "id": "Context ID --- ResponseID",
    "data_type": "Type of response: generated or written",
    "context":"List of utterances in dialogue history",
    "response": "The claim or response",
    "evidence_list": "List of evidences. Eack item in list is a list of following:"
        ["Wikipedia page Title","Wikipedia Link","Test snippet shown.","an index - not useful for the task", "optionally present value gt_evidence_added - indicates an evidence which belonged to the original utterance in WoW added for NEI claims." ],
    "response_label": "One of the three labels: SUPPORTS, REFUTES, NOT ENOUGH INFO",
    "type_label": "If the response is factual (Verifiable) or personal (Non-Verifiable)"
}
```

------------


## Results

The results for claim verification on test set. 
(Note that the results are a bit different from the first version of arxiv draft. Results in this repo are the latest.). 
![Test Results](/images/testveri.png?raw=true "Test Results")

![Validation Results](/images/validationveri.png?raw=true "Validation Results")



------------

## Citation
```
@article{gupta2021dialfact,
  title={DialFact: A Benchmark for Fact-Checking in Dialogue},
  author={Gupta, Prakhar and Wu, Chien-Sheng and Liu, Wenhao and Xiong, Caiming},
  journal={arXiv preprint arXiv:2110.08222},
  year={2021}
}
```

## Questions?
For any questions, feel free to open issues, or shoot emails to
- Jason Wu (wu.jason@salesforce.com)
- [Prakhar Gupta](https://prakharguptaz.github.io/) (CMU)

## License
The code is released under BSD 3-Clause - see [LICENSE](LICENSE.txt) for details.
