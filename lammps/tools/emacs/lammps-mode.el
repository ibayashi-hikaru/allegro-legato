;;; lammps-mode.el --- basic syntax highlighting for LAMMPS files

;; Copyright (C) 2010-18 Aidan Thompson
;; Copyright (C) 2018 Rohit Goswami

;; Author: Aidan Thompson <athomps at sandia.gov>
;; Maintainer: Rohit Goswami <r95g10 at gmail.com>
;; Created: December 4, 2010
;; Modified: July 30, 2018
;; Version: 1.5.0
;; Keywords: languages, faces
;; Homepage: https://github.com/lammps/lammps/tree/master/tools/emacs
;; Package-Requires: ((emacs "24.4"))

;; This file is not part of GNU Emacs.

;; This program is free software; you can redistribute it and/or modify
;; it under the terms of the GNU General Public License as published by
;; the Free Software Foundation; either version 2 of the License, or
;; (at your option) any later version.

;; This program is distributed in the hope that it will be useful,
;; but WITHOUT ANY WARRANTY; without even the implied warranty of
;; MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
;; GNU General Public License for more details.

;; You should have received a copy of the GNU General Public License along
;; with this program; if not, write to the Free Software Foundation, Inc.,
;; 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

;;; Commentary:
;; translation of keyword classes from tools/vim
;; see http://xahlee.org/emacs/elisp_syntax_coloring.html

;; Put this in your .emacs file to enable autoloading of lammps-mode
;; and auto-recognition of "in.*" and "*.lmp" files:
;;
;; (autoload 'lammps-mode "lammps-mode.el" "LAMMPS mode." t)
;; (setq auto-mode-alist (append auto-mode-alist
;;                               '(("in\\." . lammps-mode))
;;                               '(("\\.lmp\\'" . lammps-mode))
;;                               ))
;;

;;; Code:
 ;; define several keyword classes
(defvar lammps-output
  '("log"
    "write_restart"
    "dump"
    "undump"
    "thermo"
    "thermo_modify"
    "thermo_style"
    "print")
  "LAMMPS output.")

(defvar lammps-read
  '("include"
    "read"
    "read_restart"
    "read_data")
  "LAMMPS read.")

(defvar lammps-lattice
  '("boundary"
    "units"
    "atom_style"
    "lattice"
    "region"
    "create_box"
    "create_atoms"
    "dielectric"
    "delete_atoms"
    "change_box"
    "dimension"
    "replicate")
  "LAMMPS lattice.")

(defvar lammps-define
  '("variable"
    "group")
  "LAMMPS define.")

(defvar lammps-run
  '("minimize"
    "run")
  "LAMMPS run.")

(defvar lammps-setup
  '("min_style"
    "fix_modify"
    "run_style"
    "timestep"
    "neighbor"
    "neigh_modify"
    "fix"
    "unfix"
    "communicate"
    "newton"
    "nthreads"
    "processors"
    "reset_timestep")
  "LAMMPS setup.")

(defvar lammps-particle
  '("pair_coeff"
    "pair_style"
    "pair_modify"
    "mass"
    "velocity"
    "angle_coeff"
    "angle_style"
    "atom_modify"
    "atom_style"
    "bond_coeff"
    "bond_style"
    "delete_bonds"
    "kspace_style"
    "kspace_modify"
    "dihedral_style"
    "dihedral_coeff"
    "improper_style"
    "improper_coeff")
  "LAMMPS particle.")

(defvar lammps-repeat
  '("jump"
    "next"
    "loop")
  "LAMMPS repeat.")

(defvar lammps-operator
  '("equal"
    "add"
    "sub"
    "mult"
    "div")
  "LAMMPS operator.")

(defvar lammps-conditional
  '("if"
    "then"
    "elif"
    "else")
  "LAMMPS conditional.")

(defvar lammps-special
  '("EDGE"
    "NULL")
  "LAMMPS special.")

;; create the regex string for each class of keywords
(defvar lammps-output-regexp (regexp-opt lammps-output 'words))
(defvar lammps-read-regexp (regexp-opt lammps-read 'words))
(defvar lammps-lattice-regexp (regexp-opt lammps-lattice 'words))
(defvar lammps-define-regexp (regexp-opt lammps-define 'words))
(defvar lammps-run-regexp (regexp-opt lammps-run 'words))
(defvar lammps-setup-regexp (regexp-opt lammps-setup 'words))
(defvar lammps-particle-regexp (regexp-opt lammps-particle 'words))
(defvar lammps-repeat-regexp (regexp-opt lammps-repeat 'words))
(defvar lammps-operator-regexp (regexp-opt lammps-operator 'words))
(defvar lammps-conditional-regexp (regexp-opt lammps-conditional 'words))
(defvar lammps-special-regexp (regexp-opt lammps-special 'words))

;; Add some more classes using explicit regexp

(defvar lammps-number-regexp
  "\\<[0-9]\\>")

(defvar lammps-float-regexp
  "\\<[0-9-+]+.[0-9-+]*\\>")

(defvar lammps-comment-regexp
  "#*")

(defvar lammps-variable-regexp
  "\\$\\({[a-zA-Z0-9_]+}\\)\\|\\$[A-Za-z]")

(defvar lammps-font-lock-keywords)

;; clear memory
(setq lammps-output nil)
(setq lammps-read nil)
(setq lammps-lattice nil)
(setq lammps-define nil)
(setq lammps-run nil)
(setq lammps-setup nil)
(setq lammps-particle nil)
(setq lammps-repeat nil)
(setq lammps-operator nil)
(setq lammps-conditional nil)
(setq lammps-special nil)

;; create the list for font-lock.
;; each class of keyword is given a particular face
(setq lammps-font-lock-keywords
 `((,lammps-output-regexp . font-lock-function-name-face)
   (,lammps-read-regexp . font-lock-preprocessor-face)
   (,lammps-lattice-regexp . font-lock-type-face)
   (,lammps-define-regexp . font-lock-variable-name-face)
   (,lammps-run-regexp . font-lock-keyword-face)
   (,lammps-setup-regexp . font-lock-type-face)
   (,lammps-particle-regexp . font-lock-type-face)
   (,lammps-repeat-regexp . font-lock-string-face)
   (,lammps-operator-regexp . font-lock-warning-face)
   (,lammps-conditional-regexp . font-lock-builtin-face)
   (,lammps-special-regexp . font-lock-constant-face)
   (,lammps-float-regexp . font-lock-constant-face)
   (,lammps-number-regexp . font-lock-constant-face)
   (,lammps-comment-regexp . font-lock-constant-face)
   (,lammps-variable-regexp . font-lock-function-name-face)
	;; note: order above matters. lammps-variable-regexp� goes last because
	;; otherwise the keyword �state� in the variable �state_entry�
	;; would be highlighted.
   ))

;; define the mode
(define-derived-mode lammps-mode shell-script-mode
  "lammps mode"
  "Major mode for editing LAMMPS input scripts ..."
  ;; ...

  ;; code for syntax highlighting
  (setq font-lock-defaults '((lammps-font-lock-keywords)))

  ;; clear memory
  (setq lammps-output-regexp nil)
  (setq lammps-read-regexp nil)
  (setq lammps-lattice-regexp nil)
  (setq lammps-define-regexp nil)
  (setq lammps-run-regexp nil)
  (setq lammps-setup-regexp nil)
  (setq lammps-particle-regexp nil)
  (setq lammps-repeat-regexp nil)
  (setq lammps-operator-regexp nil)
  (setq lammps-conditional-regexp nil)
  (setq lammps-special-regexp nil)
  (setq lammps-number-regexp nil)
  (setq lammps-float-regexp nil)
  (setq lammps-comment-regexp nil)
  (setq lammps-variable-regexp nil))

(provide 'lammps-mode)
;;; lammps-mode.el ends here
