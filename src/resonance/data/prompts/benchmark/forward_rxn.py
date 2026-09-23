# prompts/forward_rxn.py

from ..base import Prompt


class ForwardRxnPrompt(Prompt):
    def get_system_prompt(self, system_prompt_option="none"):
        if system_prompt_option == "none":
            text = """
        You are an expert organic chemist specializing in the forward prediction of organic reactions.
        Your task is to predict the major product(s)
        of organic reactions given starting materials and reagents.
        """
        elif system_prompt_option == "rules":
            text = """
        You are an expert organic chemist specializing in the forward prediction of organic reactions.
        Correctly predicting the product of an organic reaction involves understanding of several key concepts.
        Some of the most important concepts to consider include:

        1. Acid-base chemistry: these reactions will proceed from a stronger acid to a weaker one.
        2. Single displacement reactions: Similarly, these reactions will proceed by a more reactive element displacing a less reactive one.
        3. Redox reactions: identify the relationship between reagents and their potential for reducing/oxidizing specific functional groups on a molecule.
        4. Mechanistic analysis: leveraging knowledge of common reaction mechanisms, one can infer key functional groups in a molecule and their potential relationships to reagents and other key intramolecular functional groups

        Your task is to predict the major product(s)
        of organic reactions given starting materials and reagents.
        """
        elif system_prompt_option == "descriptions":
            text = """
        You are an expert organic chemist specializing in the forward prediction of organic reactions.
        Correctly predicting the product of an organic reaction involves understanding of several key concepts.
        Some of the most important concepts to consider include:

        1. Acid-base chemistry: these reactions will proceed from a stronger acid to a weaker one.
        2. Single displacement reactions: Similarly, these reactions will proceed by a more reactive element displacing a less reactive one.
        3. Redox reactions: identify the relationship between reagents and their potential for reducing/oxidizing specific functional groups on a molecule.
        4. Mechanistic analysis: leveraging knowledge of common reaction mechanisms, one can infer key functional groups in a molecule and their potential relationships to reagents and other key intramolecular functional groups

        Your task is to predict the major product(s)
        of organic reactions given starting materials and reagents.

        Here are the description of some common organic reactions:

        The Stetter Reaction is a 1,4-addition (conjugate addition) of an aldehyde to an a,β-unsaturated compound, catalyzed by cyanide or a thiazolium salt. This reaction competes with the corresponding 1,2-addition, which is the Benzoin Condensation. However, the Benzoin-Condensation is reversible, and since the Stetter Reaction leads to more stable products, the main product will be derived from 1,4-addition. The mechanism of the Stetter reaction involves the cyanide ion effects an umpolung of the normal carbonyl charge affinity, and the electrophilic aldehyde carbon becomes nucleophilic after deprotonation. This in situ generated nucleophile reacts with the unsaturated compound. The thiazolium salts may be converted into ylides by deprotonation. These ylides are comparable to the cyanide adducts in their ability to catalyze the Stetter Reaction effectively.

        The Ozonolysis reaction allows the cleavage of alkene double bonds by reaction with ozone. Depending on the work up, different products may be isolated: reductive work-up gives either alcohols or carbonyl compounds, while oxidative work-up leads to carboxylic acids or ketones. The reaction mechanism of the ozonolysis reaction involves the following steps: First step is a 1,3-dipolar cycloaddition of ozone to the alkene leading to the primary ozonide (molozonide, 1,2,3-trioxolane, or Criegee intermediate) which decomposes to give a carbonyl oxide and a carbonyl compound. The carbonyl oxides are similar to ozone in being 1,3-dipolar compounds, and undergo 1,3-dipolar cycloaddition to the carbonyl compounds with the reverse regiochemistry, leading to a mixture of three possible secondary ozonides (1,2,4-trioxolanes). These secondary ozonides are more stable than primary ozonides. Even if the peroxy bridge is shielded by steric demanding groups leading to isolable products, they should not be isolated from an unmodified ozonolysis, because still more explosive side products (tetroxanes) may have been formed. Alcohols react with the carbonyl oxide to give hydroperoxy hemiacetals

        The classic Julia olefination reaction, also known as the Julia-Lythgoe olefination reaction, consists of α-deprotonation of phenyl sulfone, addition of carbanions to aldehydes and ketones, acetylation of hydroxyl groups, and elimination of sodium-amalgam desulfonation groups. consists of steps. The Julia Kocienski improvement of the Julia olefination reaction relies on the use of 1-tert-butyl-1H-tetrazol-5-yl sulfone (PT) to improve the trans stereoselectivity of olefins. The mechanism of the Julia-Kocienski olefination reaction includes the steps: metallation; addition of metallated sulfone to aldehyde, beta-alkoxysulfone adduct can occur spontaneously Smiles rearranges to form adduct, followed by beta-elimination of SO2 and aryloxy anion.

        The Oppenauer Oxidation reaction is the aluminium-catalyzed hydride shift from the α-carbon of an alcohol component to the carbonyl carbon of a second component, which proceeds over a six-membered transition state. When aldehydes or ketones are the desired products, the reaction is viewed as the Oppenauer Oxidation. Non-enolizable ketones with a relatively low reduction potential, such as benzophenone, can serve as the carbonyl component used as the hydride acceptor in this oxidation.

        The Claisen condensation is a carbon–carbon bond forming reaction that occurs between two esters or one ester and another carbonyl compound in the presence of a strong base. The reaction produces a β-keto ester or a β-diketone. - The classic Claisen condensation, a self-condensation between two molecules of a compound containing an enolizable ester. The mixed (or "crossed") Claisen condensation, where one enolizable ester or ketone and one nonenolizable ester are used. In the first step of the mechanism, an α-proton is removed by a strong base, resulting in the formation of an enolate anion, which is made relatively stable by the delocalization of electrons. Next, the carbonyl carbon of the (other) ester is nucleophilically attacked by the enolate anion. The alkoxy group is then eliminated (resulting in (re)generation of the alkoxide), and the alkoxide removes the newly formed doubly α-proton to form a new, highly resonance-stabilized enolate anion. Aqueous acid (e.g. sulfuric acid or phosphoric acid) is added in the final step to neutralize the enolate and any base still present. The newly formed β-keto ester or β-diketone is then isolated. Note that the reaction requires a stoichiometric amount of base as the removal of the doubly α-proton thermodynamically drives the otherwise endergonic reaction. That is, Claisen condensation does not work with substrates having only one α-hydrogen because of the driving force effect of deprotonation of the β-keto ester in the last step.

        The Ramberg-Bäcklund Reaction allows a base-mediated conversion of α-halosulfones into E or Z alkenes. Z alkenes are often observed with weak bases, whereas strong bases give predominantly E alkenes. Myers' modification and newer synthetic protocols allow an in situ generation of α-halosulfones, so that the readily available sulfones can be used directly without performing a specific and more problematic halogenation step. The Ramberg-Bäcklund Reaction is a convenient procedure for preparing alkenes, due to the accessibility of sulfones, the clearly defined location of the double bond as alkene rearrangements do not occur under basic conditions, and the broad range of substitution patterns that can be prepared including fully substituted alkenes, cyclobutenes and polyenes. Even polyfunctional molecules can be converted as long as base-sensitive groups are absent. The anionic mechanism for the Ramberg-Bäcklund Reaction is generally accepted. A rapid deprotonation is followed by a slow cyclization to generate a thiirane dioxide intermediate. In this step, α-iodosulfones react faster than bromo- and chlorosulfones.

        The Payne rearrangement is the isomerization, under basic conditions, of 2,3-epoxy alcohols to isomeric 1,2-epoxy alcohols with inversion of configuration. The basic mechanism of the Payne rearrangement involves deprotonation of the free hydroxyl group, invertive nucleophilic attack on the proximal epoxide carbon, and re-protonation of the newly freed alkoxide. The Payne rearrangement occurs with inversion of stereochemistry at C-2. Substrates containing multiple adjacent hydroxyl groups may undergo "cascade" epoxide migrations with inversion at each site of nucleophilic attack. In one example, inversion of three contiguous stereocenters results after two epoxide migrations, opening of the epoxide by carboxylate, and hydrolysis of the resulting lactone.

        The Seyferth-Gilbert Homologation is the base-promoted reaction of dimethyl (diazomethyl)phosphonate with aldehydes and aryl ketones at low temperatures, and provides a synthesis of alkynes. The Ohira-Bestmann modification using dimethyl 1-diazo-2-oxopropylphosphonate allows the conversion of base-labile substrates such as enolizable aldehydes, which would tend to undergo aldol condensation under the Seyferth-Gilbert conditions. The mechanism of the Seyferth-Filber homologation reaction involves the following: The deprotonated Seyferth-Gilbert reagent adds to the carbonyl compound forming an alkoxide that closes to give an oxaphosphetane. Comparable to the Wittig Reaction, a cycloelimination yields a stable dimethyl phosphate anion and a diazoalkene. Upon warming of the reaction mixture to room temperature, loss of nitrogen gives a vinylidene carbene that yields the desired alkyne after 1,2-migration of one of the substituents. For substrates with substituents that have low migratory aptitudes, such as 4-nitrobenzaldehyde, the existence of the nitrene can be demonstrated by adding an excess of amine or alcohol to the reaction mixture, which yields the products of insertion reactions.

        The Hofmann rearrangement is the organic reaction of a primary amide to a primary amine with one less carbon atom. The reaction involves oxidation of the nitrogen followed by rearrangement of the carbonyl and nitrogen to give an isocyanate intermediate. The reaction of bromine with sodium hydroxide forms sodium hypobromite in situ, which transforms the primary amide into an intermediate isocyanate. The formation of an intermediate nitrene is not possible because it implies also the formation of a hydroxamic acid as a byproduct, which has never been observed. The intermediate isocyanate is hydrolyzed to a primary amine, giving off carbon dioxide and producing the final primary amine product.

        The Wohl Ziegler Bromination reaction leads to the bromination of allylic position susing N-bromosuccinimide via a radical pathway. It is very important to keep the concentration of Br2 and HBr low to prevent side reactions derived from simple ionic addition with the alkene. These reagents are therefore generated in situ from NBS. The catalytically active species is Br2, which is almost always present in NBS samples (red colour). A radical initiator (UV, AIBN) is needed for the homolytic bond cleavage of Br2. The allylic position is favoured for hydrogen abstraction, because the resulting radical intermediate is resonance stabilized. The first step, bromination via the radical carbanion, is favored to occur at the more highly substituted position because the corresponding intermediate radicals are better stabilized. CCl4 is the solvent of choice, because NBS is poorly soluble and resulting succinimide is insoluble and floats at the surface. This keeps the concentration of reagents low and is a signal that the reaction is finished. However, environmental concerns have all but eliminated the use of CCl4, and its replacement, CH2Cl2, is being restricted as well. Many other solvents are reactive toward NBS, and are thus unsuitable, but acetonitrile can be used to good effect.

        The Rubuttom oxidation reaction is the synthesis of α-hydroxy ketones is achieved by reaction of silyl enol ethers with mCPBA, with subsequent rearrangement. Aqueous work up or reaction with TBAF (fluoride ions) gives the desired product after desilylation. The enol ether double bond is epoxidized by the peracid. Relief of the epoxide ring strain drives the rearrangement with migration of the silyl group to give the silylated α-hydroxy ketone product.

        The Glaser Coupling is a synthesis of symmetric or cyclic bisacetylenes via a coupling reaction of terminal alkynes. Mechanistically, the reaction is similar to the Eglinton Reaction; the difference being the use of catalytic copper(I), which is reoxidized in the catalytic cycle by oxygen in the reaction medium.

        The Achmatowicz rearrangement reaction conversta a furan to a dihydropyran. A furfurly alcohol is reacted with bromine in methanol to to 2,5-dimethoxy-2,5-dihydrofuran which rearranges to the dihydropyran with dilute sulfuric acid. Additional reaction steps, alcohol protection with methyl orthoformate and boron trifluoride) and then ketone reduction with sodium borohydride produce an intermediate from which many monosaccharides can be synthesised.

        The Acyloin condensation reaction is the bimolecular reductive coupling of carboxylic esters by reaction with metallic sodium in an inert solvent under reflux gives an α-hydroxyketone. This reaction is favored when R is an alkyl. With longer alkyl chains, higher boiling solvents can be used. The intramolecular version of this reaction has been used extensively to close rings of different sizes, e.g. paracyclophanes or catenanes. If the reaction is carried out in the presence of a proton donor, such as alcohol, simple reduction of the ester to the alcohol takes place. The Benzoin Condensation produces similar products, although with aromatic substituents and under different conditions. When the acyloin condensation is carried out in the presence of chlorotrimethylsilane, the enediolate intermediate is trapped as the bis-silyl derivative. This can be isolated and subsequently is hydrolysed under acidic condition to the acyloin, which gives a better overall yield.

        The Aza-claisen rearrangement or aliphatic Claisen Rearrangement is a 3,3 sigmatropic rearrangement in which an allyl vinyl ether is converted thermally to an unsaturated carbonyl compound. The aromatic Claisen Rearrangement is accompanied by a rearomatization. The etherification of alcohols or phenols and their subsequent Claisen Rearrangement under thermal conditions makes possible an extension of the carbon chain of the molecule. The Claisen Rearrangement may be viewed as the oxa-variant of the Cope Rearrangement. The reaction proceeds preferably via a chair transition state. Chiral, enantiomerically enriched starting materials give products of high optical purity. A boat transition state is also possible, and can lead to side products. The aromatic Claisen Rearrangement is followed by a rearomatization. When the ortho position is substituted, rearomatization cannot take place. The allyl group must first undergo a Cope Rearrangement to the para-position before tautomerization is possible.

        The Reductive etherification reaction is a one-pot organic synthesis reaction that converts aldehydes or ketones into ethers using reducing agents like triethylsilane or hydrogen gas with a catalyst. The reaction generally proceeds via the following mechanism: The catalyst activates the carbonyl compound and an intermediary hemiacetal or enol ether is formed. The reducing agent (like silane) produces the final ether product.

        The Stobbe condensation reaction is a reaction of an aldehyde or ketone with an ester of succinic acid to generate alkylidene succinic acid or related derivatives. The mechanism of the Stobbe condensation proceeds through a sequence of addition, lactonization, and ring-opening steps. The reaction proceeds via the following general steps: Deprotonation and Enolate Formation: In the presence of a strong base (e.g., alkoxides, NaH, or t-BuOK), the α-hydrogen of diethyl succinate is abstracted, forming a stabilized enolate anion. Aldol-Type Addition: The nucleophilic enolate attacks the electrophilic carbonyl carbon of an aldehyde or ketone, leading to a tetrahedral alkoxide intermediate. Lactonization: The alkoxide intramolecularly attacks the adjacent ester carbonyl group, displacing ethoxide to form a five-membered γ-lactone ring. This step is reversible, but the subsequent reaction drives the equilibrium forward. Ring Opening: Under basic conditions, the γ-lactone undergoes hydrolysis and elimination, resulting in ring opening to produce a carboxylate salt. The formation of this resonance-stabilized anion is a key thermodynamic driving force. Protonation: Acidic workup converts the carboxylate into the final mono-ester product, an alkylidene succinic acid derivative.

        The Acetoacetic Ester synthesis reaction is a reaction where ethylacetoacetate is alkylated at the alpha-carbon position to both carbonyl groups and is then converted into an alpha-substituted acetone. The reaction proceeds via the following steps: A strong base deprotonates the dicarbonyl α-carbon. This carbon is preferred over the methyl carbon because the formed enolate is conjugated and thus resonance stabilized. The carbon then undergoes nucleophilic substitution. When heated with aqueous acid, the newly alkylated ester is hydrolyzed to a β-keto acid, which is decarboxylated to form a methyl ketone. The alkylated ester can undergo a second substitution to produce the dialkylated product

        The Tiffeneau-Demjanov rearrangement is often used to transform a cyclic ketone into a homologue that is one ring size larger. This reaction proceeds with a cyanohydrin intermediate.

        The Arndt-Eistert Synthesis allows the formation of homologated carboxylic acids or their derivatives by reaction of the activated carboxylic acids with diazomethane and subsequent Wolff-Rearrangement of the intermediate diazoketones in the presence of nucleophiles such as water, alcohols, or amines. In the first step of this one-carbon homologation, the diazomethane carbon is acylated by an acid chloride or mixed anhydride, to give an α-diazoketone. The excess diazomethane can be destroyed by addition of small amounts of acetic acid or vigorous stirring. Most α-diazoketones are stable and can be isolated and purified by column chromatography (see recent literature for specific methods). The key step of the Arndt-Eistert Homologation is the Wolff-Rearrangement of the diazoketones to ketenes, which can be accomplished thermally, photochemically or by silver(I) catalysis. The reaction is conducted in the presence of nucleophiles such as water (to yield carboxylic acids), alcohols (to give esters) or amines (to give amides), to capture the ketene intermediate and avoid the competing formation of diketenes.

        The Eschweiler-Clarke Methylation reaction allows the preparation of tertiary methylamines from secondary amines via treatment with formaldehyde in the presence of formic acid. The formate anion acts as hydride donor to reduce the imine or iminium salt, so that the overall process is a reductive amination. The formation of quaternary amines is not possible.

        The Riley oxidation is a selenium dioxide-mediated oxidation of methylene groups adjacent to carbonyls. The oxidation begins with attack by the enol tautomer at the electrophilic selenium center. Then, a molecule of water is lost, and a second equivalent of water attacks the alpha position. In the final step, selenic acid is released to give the 1,2 dicarbonyl product

        The Dess-Martin oxidation reaction leads to the oxidation of primary alcohols ot aldehydes and secondary alcohols ot ketons using Dess-Martin periodinane. The reaction relies on this hypervalent iodine reaction to selectively and mildly oxidize alcohols or ketones.

        The Hunsdiecker reaction involves the reaction of silver(I) salts of carboxylic acids with halogens to give unstable intermediates which readily decarboxylate thermally to yield alkyl halides. The reaction is believed to involve homolysis of the C-C bond and a radical chain mechanism. The Cristol-Firth modification of the Hunsdiecker reaction uses a carboxylic acid and mercury(II) oxide in the presence of a halogen (Br2) as reagents instead of the anhydrous silver salts.

        """
        return text

    def get_task_template(self, representation="smiles", user_prompt_option="icl"):
        # representation name
        rep_names = {
            "smiles": "SMILES",
            "atom_mapped": "atom-mapped SMILES",
            "iupac": "IUPAC names",
        }

        # atom-mapping note
        rep_notes = {
            "smiles": "",
            "atom_mapped": "The numbers after colons are atom mapping numbers.",
            "iupac": "",
        }

        # step-by-step instructions
        rep_instructions = {
            "smiles": """Instructions:
        1. Identify the functional groups in the starting material
        2. Determine which reagent(s) will react with these functional groups
        3. Apply your knowledge of organic reaction mechanisms to predict the major product
        4. Consider regioselectivity and stereoselectivity where applicable
        If your proposed mechanism or product seems chemically implausible, revise your
        reasoning and try a different mechanistic path.
""",
            "atom_mapped": """Before giving your final product, reason step-by-step through
        the mechanism using the provided atom mappings:
        1. Identify the most plausible mechanistic class.
        2. Explicitly describe each mechanistic step as a sequence of electron movements, using arrow-pushing logic.
            - Electrons can originate from lone pairs or bonds (electron sources)
            - Electrons can move toward atoms or bonds that can accept them (electron sinks),
            - Refer to atoms and bonds using the atom mapping numbers in the SMILES
        3. Continue until a reasonable product structure is reached
        If your proposed mechanism or product seems chemically implausible, revise your
        reasoning and try a different mechanistic path.
""",
            "iupac": """Instructions:
1. Identify the functional groups in the starting material
2. Determine which reagent(s) will react with these functional groups
3. Apply your knowledge of organic reaction mechanisms to predict the major product
If your proposed mechanism or product seems chemically implausible, revise your reasoning and try a different mechanistic path.
""",
        }

        rep_name = rep_names[representation]
        rep_note = rep_notes[representation]
        rep_instruction = rep_instructions[representation]

        example_icl = ""
        if user_prompt_option in ["icl_random", "icl_specific"]:
            icl_placeholder = f"{example_icl}\n\n"
        else:
            icl_placeholder = ""

        text = """{}Here are the {} of the starting material and reagent(s).
Starting material: {{starting_material}}
Reagents: {{reagent_list}}
{}
Predict the single most likely primary product of the chemical reaction in {} format.
The reaction should be chemically reasonable, following typical organic reactivity principles.
{}
Return the answer between <answer> and </answer> tags.
""".format(
            icl_placeholder,
            rep_name,
            rep_note,
            rep_name,
            rep_instruction,
        )

        return text
